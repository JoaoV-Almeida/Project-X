import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import os
import json
from pathlib import Path
from datetime import datetime, time

# ============================================
# 1. CONFIGURAÇÃO & ESTILOS
# ============================================
st.set_page_config(page_title="Flight Risk AI", page_icon="✈️", layout="wide")

st.markdown("""
    <style>
    .risk-card {
        padding: 20px; border-radius: 12px; color: white;
        text-align: center; margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button { width: 100%; font-weight: bold; border-radius: 8px; height: 3em; }
    </style>
    """, unsafe_allow_html=True)

# ============================================
# 2. CARREGAMENTO DE ARTEFATOS
# ============================================
@st.cache_resource
def load_artifacts():
    base_path = os.path.join(os.path.dirname(__file__), "flight_risk_app")
    try:
        model = joblib.load(os.path.join(base_path, "flight_risk_model.pkl"))
        scaler = joblib.load(os.path.join(base_path, "flight_risk_scaler.pkl"))
        encoder = joblib.load(os.path.join(base_path, "flight_risk_encodings.pkl"))
        holidays = joblib.load(os.path.join(base_path, "flight_holidays.pkl"))
        traffic_stats = joblib.load(os.path.join(base_path, "flight_traffic_stats.pkl"))
        dist_lookup = joblib.load(os.path.join(base_path, "flight_distance_lookup.pkl"))
        return model, scaler, encoder, set(holidays), traffic_stats, dist_lookup
    except Exception as e:
        st.error(f"❌ Erro ao carregar arquivos: {e}")
        return None, None, None, None, None, None

model, scaler, encoder, holidays, traffic_stats, dist_lookup = load_artifacts()

@st.cache_data
def load_airports_data():
    airports_path = Path(__file__).parent / "data" / "airports.json"
    try:
        with open(airports_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except: return []

airports_data = load_airports_data()

# ============================================
# 3. MOTOR DE PREDIÇÃO (CORRIGIDO PARA 20 COLUNAS)
# ============================================
def predict_flight_risk(origin, dest, date_obj, time_obj):
    sched_hour = time_obj.hour
    week_of_year = date_obj.isocalendar()[1]
    month, day_of_week = date_obj.month, date_obj.weekday()
    
    is_holiday = 1 if date_obj in holidays else 0
    is_near_holiday = 1 if any(abs((h - date_obj).days) <= 3 for h in holidays) else 0

    # Trigonométricas
    hour_sin, hour_cos = np.sin(2 * np.pi * sched_hour / 24), np.cos(2 * np.pi * sched_hour / 24)
    month_sin, month_cos = np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12)
    day_sin, day_cos = np.sin(2 * np.pi * day_of_week / 7), np.cos(2 * np.pi * day_of_week / 7)

    distance = dist_lookup.get((origin.lower(), dest.lower()), 800)
    
    # CRIANDO AS 20 COLUNAS EXATAS QUE O ENCODER ESPERA
    # A ordem e os nomes devem ser IDÊNTICOS ao X_train do notebook
    input_df = pd.DataFrame([{
        'distance': distance,
        'week_of_year': week_of_year,
        'month_sin': month_sin,
        'month_cos': month_cos,
        'dayofweek_sin': day_sin,
        'dayofweek_cos': day_cos,
        'hour_sin': hour_sin,
        'hour_cos': hour_cos,
        'is_weekend': 1 if day_of_week >= 5 else 0,
        'is_holiday': is_holiday,
        'is_near_holiday': is_near_holiday,
        'dep_hour': sched_hour,
        'origin': origin,
        'dest': dest,
        'airport_route': f"{origin} -> {dest}",
        'tail_number': 'UNKNOWN',
        'operating_airline': 'UNKNOWN',
        'deptimeblk': f"{sched_hour:02d}00-{sched_hour:02d}59",
        'distance_category': 'medium', # Coluna 19
        'is_morning': 1 if 5 <= sched_hour <= 11 else 0 # Coluna 20
    }])

    # 1. Target Encoding (Agora com 20 colunas, não dará erro)
    input_encoded = encoder.transform(input_df)
    
    # 2. Scaling (Apenas colunas numéricas)
    num_cols = ['distance', 'week_of_year', 'month_sin', 'month_cos', 'dayofweek_sin', 
                'dayofweek_cos', 'hour_sin', 'hour_cos', 'is_weekend', 'is_holiday', 'is_near_holiday', 'dep_hour']
    input_encoded[num_cols] = scaler.transform(input_encoded[num_cols])
    
    # 3. Predict
    prob = model.predict_proba(input_encoded)[:, 1][0]
    avg_delay = traffic_stats['route_means'].get(f"{origin}_{dest}", 15)
    
    return prob, avg_delay

# ============================================
# 4. UI SIDEBAR & MAIN
# ============================================
st.sidebar.header("✈️ Flight Search")
unique_cities = sorted(list(set([a['city_name'] for a in airports_data])))
selected_city = st.sidebar.selectbox("Select City", options=[""] + unique_cities)

if selected_city:
    available_origins = sorted([f"{a['airport_code']} - {a['airport_name']}" for a in airports_data if a['city_name'] == selected_city])
    selected_origin_full = st.sidebar.selectbox("Origin Airport", options=available_origins)
    origin_code = selected_origin_full.split(" - ")[0]
else: origin_code = None

if origin_code:
    route_keys = [k for k in dist_lookup.keys() if k != 'global_default']
    possible_dests = sorted(list(set([k[1].upper() for k in route_keys if k[0].upper() == origin_code])))
    dest_options = [f"{d} - {next((a['airport_name'] for a in airports_data if a['airport_code'] == d), d)}" for d in possible_dests]
    selected_dest_full = st.sidebar.selectbox("Destination Airport", options=dest_options)
    dest_code = selected_dest_full.split(" - ")[0] if selected_dest_full else None
else: dest_code = None

travel_date = st.sidebar.date_input("Date", value=datetime.today())
flight_time = st.sidebar.time_input("Time", value=time(9, 0))

st.title("🛫 Flight Risk AI")

if "result" not in st.session_state: st.session_state.result = None

if origin_code and dest_code:
    if st.button("🔍 ANALYZE RISK", type="primary"):
        with st.spinner("Analyzing..."):
            prob, avg_delay = predict_flight_risk(origin_code, dest_code, travel_date, flight_time)
            
            # Tiers validados
            if prob >= 0.75: risk, color, note = "HIGH", "#dc3545", "42.1% Historical Delay Rate"
            elif prob >= 0.35: risk, color, note = "MEDIUM", "#ffc107", "22.1% Historical Delay Rate"
            else: risk, color, note = "LOW", "#28a745", "11.2% Historical Delay Rate"

            st.session_state.result = {"prob": prob, "class": risk, "color": color, "note": note}

if st.session_state.result:
    res = st.session_state.result
    st.markdown(f'<div class="risk-card" style="background-color:{res["color"]};"><h1>{res["class"]} RISK</h1><h3>Prob: {res["prob"]:.1%}</h3><p>{res["note"]}</p></div>', unsafe_allow_html=True)