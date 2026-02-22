import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import joblib
import google.generativeai as genai

from personas import PERSONAS

# ------------------------
# GEMINI SETUP
# ------------------------
api_key = None
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
elif "gemini" in st.secrets and "api_key" in st.secrets["gemini"]:
    api_key = st.secrets["gemini"]["api_key"]

if api_key:
    try:
        genai.configure(api_key=api_key)
        llm_model = genai.GenerativeModel("gemini-1.5-flash") # Using 1.5-flash for better compatibility
    except Exception as e:
        st.error(f"❌ Error configuring Gemini API: {e}")
        llm_model = None
else:
    st.warning("⚠️ Gemini API key not configured in secrets.toml or Streamlit Cloud Secrets. AI Analysis will be disabled.")
    llm_model = None

# ------------------------
# PAGE CONFIG 
# ------------------------
st.set_page_config(page_title="Flight Risk AI", page_icon="✈️", layout="wide")

# ------------------------
# LOAD DATABASES & ML ARTIFACTS
# ------------------------
@st.cache_resource
def load_ml_pipeline():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TARGET_DIR = os.path.join(BASE_DIR, "flight_risk_app")
    
    try:
        model = joblib.load(os.path.join(TARGET_DIR, "flight_risk_model.pkl"))
        scaler = joblib.load(os.path.join(TARGET_DIR, "flight_risk_scaler.pkl"))
        encoder = joblib.load(os.path.join(TARGET_DIR, "flight_risk_encodings.pkl"))
        dist_lookup = joblib.load(os.path.join(TARGET_DIR, "flight_distance_lookup.pkl"))
        holidays = joblib.load(os.path.join(TARGET_DIR, "flight_holidays.pkl"))
        
        traffic_path = os.path.join(TARGET_DIR, "traffic_stats.pkl")
        if not os.path.exists(traffic_path):
            traffic_path = os.path.join(TARGET_DIR, "flight_traffic_stats.pkl")
        traffic = joblib.load(traffic_path)
        
        return model, scaler, encoder, dist_lookup, holidays, traffic
        
    except Exception as e:
        st.error(f"Failed to load ML artifacts. Error: {e}")
        return None, None, None, {}, [], None

@st.cache_data
def load_airport_db():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    AIRPORTS_PATH = os.path.join(BASE_DIR, "data", "airports.json")
    try:
        with open(AIRPORTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return {item["airport_code"].lower(): f"{item['city_name']} - {item['airport_name']} ({item['airport_code']})" for item in data}
    except FileNotFoundError:
        st.warning(f"airports.json not found at {AIRPORTS_PATH}. Using raw airport codes.")
        return {}

ml_model, scaler, encoder, dist_lookup, holidays_list, traffic_stats = load_ml_pipeline()
airport_map = load_airport_db()

# ------------------------
# ML PREDICTION WRAPPER 
# ------------------------
def predict_delay_with_ml(origin, dest, date_obj, dep_dt, airline):
    if ml_model is None:
        return random.randint(10, 90)

    origin_lower = origin.lower()
    dest_lower = dest.lower()
    dep_hour = dep_dt.hour
    
    dist = dist_lookup.get((origin_lower, dest_lower), 1000) 
    is_holiday = 1 if date_obj in holidays_list else 0
    is_near_holiday = 1 if (date_obj + timedelta(days=2)) in holidays_list or (date_obj - timedelta(days=2)) in holidays_list else 0

    num_data = {
        'distance': dist,
        'week_of_year': date_obj.isocalendar()[1],
        'month_sin': np.sin(2 * np.pi * date_obj.month / 12.0),
        'month_cos': np.cos(2 * np.pi * date_obj.month / 12.0),
        'dayofweek_sin': np.sin(2 * np.pi * date_obj.weekday() / 7.0),
        'dayofweek_cos': np.cos(2 * np.pi * date_obj.weekday() / 7.0),
        'hour_sin': np.sin(2 * np.pi * dep_hour / 24.0),
        'hour_cos': np.cos(2 * np.pi * dep_hour / 24.0),
        'is_weekend': 1 if date_obj.weekday() >= 5 else 0,
        'is_holiday': is_holiday,
        'is_near_holiday': is_near_holiday,
        'dep_hour': dep_hour
    }
    df_num = pd.DataFrame([num_data])

    cat_data = {
        'tail_number': 'UNKNOWN', 
        'origin': origin_lower,
        'dest': dest_lower,
        'airport_route': f"{origin_lower}_{dest_lower}",
        'deptimeblk': f"{dep_hour:02d}00-{dep_hour:02d}59",
        'operating_airline': airline
    }
    df_cat = pd.DataFrame([cat_data])

    if traffic_stats:
        route_key = f"{origin_lower}_{dest_lower}"
        global_val = traffic_stats.get('global_mean', 0)
        df_traffic = pd.DataFrame([{
            'route_mean': traffic_stats.get('route_means', {}).get(route_key, global_val),
            'origin_mean': traffic_stats.get('origin_means', {}).get(origin_lower, global_val)
        }])
    else:
        df_traffic = pd.DataFrame([{'route_mean': 0, 'origin_mean': 0}])

    X_raw = pd.concat([df_num, df_cat, df_traffic], axis=1)

    if encoder:
        X_encoded = encoder.transform(X_raw)
    else:
        X_encoded = X_raw

    if scaler and hasattr(scaler, "feature_names_in_"):
        scaler_cols = scaler.feature_names_in_
        for col in scaler_cols:
            if col not in X_encoded.columns:
                X_encoded[col] = 0
        X_encoded[scaler_cols] = scaler.transform(X_encoded[scaler_cols])

    X_final = X_encoded.copy()
    if hasattr(ml_model, "feature_names_in_"):
        expected_cols = ml_model.feature_names_in_
        for col in expected_cols:
            if col not in X_final.columns:
                X_final[col] = 0
        X_final = X_final[expected_cols]

    try:
        if hasattr(ml_model, "predict_proba"):
            prob_delay = ml_model.predict_proba(X_final)[0][1] 
            return round(prob_delay * 100, 1) 
        else:
            return 50.0
    except Exception as e:
        return 0.0

# ------------------------
# SYNTHETIC SCHEDULE GENERATOR 
# ------------------------
@st.cache_data
def generate_route_flights(origin, dest, date):
    airlines = ["DL", "AA", "UA", "WN", "B6"] 
    flights = []
    base_time = datetime.combine(date, datetime.min.time()) + timedelta(hours=6)
    
    dist = dist_lookup.get((origin.lower(), dest.lower()), 1000) if dist_lookup else 1000
    flight_duration_mins = int((dist / 500) * 60) + 45 
    
    for _ in range(random.randint(4, 7)):
        airline = random.choice(airlines)
        dep_dt = base_time + timedelta(minutes=random.randint(0, 720))
        arr_dt = dep_dt + timedelta(minutes=flight_duration_mins)
        
        probability = predict_delay_with_ml(origin, dest, date, dep_dt, airline)
            
        flights.append({
            "flight_id": f"{airline}-{random.randint(100, 999)}",
            "airline": airline,
            "origin": origin.upper(),
            "dest": dest.upper(),
            "depart_time": dep_dt.strftime("%I:%M %p"),
            "arrival_time": arr_dt.strftime("%I:%M %p"),
            "duration": f"{flight_duration_mins // 60}h {flight_duration_mins % 60}m",
            "delay_probability": probability
        })
    return sorted(flights, key=lambda x: datetime.strptime(x['depart_time'], "%I:%M %p"))

# ------------------------
# STATE MANAGEMENT
# ------------------------
if 'current_view' not in st.session_state:
    st.session_state.current_view = "search"
if 'selected_flight' not in st.session_state:
    st.session_state.selected_flight = None
if 'saved_persona' not in st.session_state: 
    st.session_state.saved_persona = "executive"
if 'messages' not in st.session_state:
    st.session_state.messages = []

# 🌟 NEW: Resets the chat history and opens the flight details
def view_flight_details(flight, selected_persona):
    st.session_state.selected_flight = flight
    st.session_state.saved_persona = selected_persona
    st.session_state.current_view = "details"
    
    # Reset Chat for the new flight
    st.session_state.messages = []
    if "gemini_chat" in st.session_state:
        del st.session_state.gemini_chat

def go_back_to_search():
    st.session_state.current_view = "search"

# ------------------------
# UI: TOP NAVIGATION & SEARCH
# ------------------------
valid_routes = list(dist_lookup.keys()) if dist_lookup else []
valid_origins = sorted(list(set([route[0] for route in valid_routes]))) if valid_routes else ["jfk", "lax", "ord"]

st.title("✈️ AI Flight Risk Predictor & Concierge")

if st.session_state.current_view == "search":
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        search_date = st.date_input("Date", min_value=datetime.today())
    with col2:
        origin_input = st.selectbox("Origin", options=valid_origins, format_func=lambda x: airport_map.get(x, x.upper()), index=valid_origins.index('jfk') if 'jfk' in valid_origins else 0)
    with col3:
        valid_dests = sorted([route[1] for route in valid_routes if route[0] == origin_input]) if valid_routes else ["lax", "ord", "jfk"]
        dest_input = st.selectbox("Destination", options=valid_dests, format_func=lambda x: airport_map.get(x, x.upper()), index=0)
    with col4:
        persona_key = st.selectbox("Traveler Profile", list(PERSONAS.keys()), format_func=lambda x: PERSONAS[x]["label"])

    st.button("Search Flights", type="primary", use_container_width=True)
    st.markdown("---")

    # --- MAIN VIEW: LIST OF FLIGHTS ---
    if origin_input and dest_input:
        flights = generate_route_flights(origin_input, dest_input, search_date)
        
        st.subheader(f"Available Flights: {origin_input.upper()} ➔ {dest_input.upper()}")
        
        if not flights:
            st.warning("No flights found for this route.")
            
        for f in flights:
            prob = f["delay_probability"]
            if prob < 20: flag, status = "🟢", "Low Probability"
            elif prob < 30: flag, status = "🟡", "Moderate Probability"
            else: flag, status = "🔴", "High Probability"

            with st.container(border=True):
                c1, c2, c3, c4 = st.columns([1.5, 2.5, 2, 1], vertical_alignment="center")
                c1.markdown(f"**{f['airline']}**<br>{f['flight_id']}", unsafe_allow_html=True)
                c2.markdown(f"**{f['depart_time']}** ➔ **{f['arrival_time']}**<br><small>{f['duration']} direct</small>", unsafe_allow_html=True)
                c3.markdown(f"**Risk:** {flag} {status}<br><small>{prob}% Chance of Delay</small>", unsafe_allow_html=True)
                c4.button("Select Flight", key=f['flight_id'], on_click=view_flight_details, args=(f, persona_key))

# ------------------------
# UI: DETAILED ITINERARY VIEW (CHATBOT MODE)
# ------------------------
elif st.session_state.current_view == "details":
    flight = st.session_state.selected_flight
    active_persona_key = st.session_state.saved_persona
    persona = PERSONAS[active_persona_key] 
    
    st.button("← Back to Search Results", on_click=go_back_to_search)
    
    st.markdown(f"""
    <div style="background-color: #1E1E1E; padding: 20px; border-radius: 10px; border-left: 5px solid #4CAF50; margin-bottom: 20px;">
        <h2 style="margin-top: 0px;">Flight Itinerary: {flight['origin']} to {flight['dest']}</h2>
        <div style="display: flex; justify-content: space-between;">
            <div><strong>Airline:</strong> {flight['airline']} ({flight['flight_id']})</div>
            <div><strong>Departure:</strong> {flight['depart_time']}</div>
            <div><strong>Arrival:</strong> {flight['arrival_time']}</div>
            <div><strong>Duration:</strong> {flight['duration']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    left_col, right_col = st.columns([1, 1.2])

    with left_col:
        st.subheader("📊 ML Risk Assessment")
        prob = flight["delay_probability"]
        
        if prob < 20: 
            st.success(f"**Low Risk:** There is only a {prob}% chance of this flight being delayed.")
            status_level = "Low Risk"
        elif prob < 30: 
            st.warning(f"**Moderate Risk:** There is a {prob}% chance of this flight being delayed.")
            status_level = "Moderate Risk"
        else: 
            st.error(f"**High Risk:** There is a {prob}% chance of this flight being delayed.")
            status_level = "High Risk"
            
        st.markdown("---")
        st.subheader("👤 Traveler Profile")
        st.markdown(f"**Archetype:** {persona.get('label', 'Traveler')}")
        st.markdown(f"**Opportunity Cost:** {persona.get('opportunity_cost', 'N/A')}")
        st.markdown(f"**Behavior:** {persona.get('customer_behavior', 'N/A')}")
        st.markdown(f"**Flight Preferences:** {persona.get('flight_preferences', 'N/A')}")
        st.markdown(f"**Coping Style:** {persona.get('coping_style', 'N/A')}")
        if "insurance" in persona:
            st.markdown(f"**Recommended Insurance:** {persona['insurance'].get('name', 'General')} ({persona['insurance'].get('focus', '')})")

    # ---------------------------------------------------------
    # 🤖 THE NEW INTERACTIVE CHATBOT PANEL
    # ---------------------------------------------------------
    with right_col:
        st.subheader(f"🤖 Concierge Chat ({persona.get('label')})")
        
        # 1. Initialize the Chat Session and the Auto-Greeting
        if "gemini_chat" not in st.session_state:
            st.session_state.gemini_chat = llm_model.start_chat(history=[])
            
            insurance_data = persona.get('insurance', {})
            
            # The secret system prompt that kicks off the chat
            initial_prompt = f"""
            You are the AI Flight Risk Concierge. 
            Traveler Archetype: {persona.get('label')}
            Tone Instruction: {persona.get('tone')}

            Flight Situation: {prob}% delay risk ({status_level}). {flight['origin']} to {flight['dest']} on {flight['airline']}.
            
            Psychology: 
            - Opportunity Cost: {persona.get('opportunity_cost')}
            - Behavior: {persona.get('customer_behavior')}
            - Preferences: {persona.get('flight_preferences')}
            - Coping Style: {persona.get('coping_style')}

            Task: Actively introduce yourself to the traveler in character. 
            Provide:
            1. A brief assessment of the {prob}% delay risk.
            2. Immediate advice on what they should do right now.
            3. A seamless pitch for "{insurance_data.get('name')}" using this exact logic: "{insurance_data.get('recommendation_logic')}".
            Ask them if they need any alternative flights or lounge info!
            """
            
            with st.spinner("Concierge is preparing your strategy..."):
                try:
                    response = st.session_state.gemini_chat.send_message(initial_prompt)
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
                except Exception as e:
                    st.error(f"API Error: {e}")

        # 2. Render Chat History in a clean container
        chat_container = st.container(height=450)
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # 3. Chat Input Box (The user talks back)
        if user_query := st.chat_input("Ask your concierge a question..."):
            
            # Show user message instantly
            st.session_state.messages.append({"role": "user", "content": user_query})
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(user_query)
                
                # Fetch AI response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        # Reminder to keep the AI in character during the back-and-forth
                        reminder = f"(Reminder: Stay strictly in character as the Concierge for the {persona.get('label')} traveler. Keep it concise.)\n\nTraveler says: {user_query}"
                        try:
                            reply = st.session_state.gemini_chat.send_message(reminder)
                            st.markdown(reply.text)
                            st.session_state.messages.append({"role": "assistant", "content": reply.text})
                        except Exception as e:
                            st.error(f"Error: {e}")