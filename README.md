Aethra I 🛫

Intelligent prediction of flight delay and cancellation risks

A Streamlit application that uses Machine Learning (XGBoost) and Generative AI (Google Gemini) to analyze flight routes and provide personalized recommendations based on 10 traveler personas.

🎯 Features

Predictive Risk Analysis: Forecasts delay probability based on historical route patterns

10 Traveler Personas: Customized recommendations (Executive, Student, Retiree, Tourist, Digital Nomad, etc.)

Integrated AI Chatbot: Conversational assistant powered by Google Gemini for personalized guidance

Opportunity Cost Analysis: Calculates the financial impact of delays for each persona

Insurance Quotes: Dynamic pricing based on risk level

Cascading Filters: Intuitive selection flow — City → Departure Airport → Destination

391 Real Airports: Complete dataset with real U.S. airport names

🚀 How to Run
Prerequisites

Python 3.13+

Google Gemini API key (configured in secrets.toml)

Method 1: Preconfigured Virtual Environment
c:/Users/Juliano.jcs/dev/Project-X/.venv/Scripts/python.exe -m streamlit run app.py
Method 2: Activating the Virtual Environment
# Windows
.\.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate

# Run application
streamlit run app.py
Method 3: Fresh Installation
# Create virtual environment
python -m venv .venv

# Activate environment (Windows)
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py
🔑 Gemini API Configuration

Create a secrets.toml file in the project root:

[gemini]
api_key = "your-api-key-here"

Get your key at: https://aistudio.google.com/apikey

📊 Project Structure
Project-X/
├── app.py                          # Main Streamlit application
├── data/
│   └── airports.json              # 391 airports with real names
├── flight_risk_app/
│   ├── flight_risk_model.json     # Trained XGBoost model
│   ├── flight_risk_scaler.pkl     # StandardScaler
│   ├── flight_risk_encodings.pkl  # Label encodings
│   ├── flight_distance_lookup.pkl # Route distances
│   └── flight_traffic_stats.pkl   # Traffic statistics
├── scripts/
│   ├── extract_airports.py        # Airport data extraction
│   ├── debug_cities.py            # City filter tests
│   ├── test_filters.py            # Filter validation
│   └── test_santa_barbara.py      # Specific route tests
├── requirements.txt               # Python dependencies
├── secrets.toml                   # Gemini API key (not versioned)
└── README.md                      # This file
🧠 Technologies Used

Streamlit 1.53.1 — UI framework

XGBoost 3.1.3 — Machine learning model

scikit-learn 1.8.0 — Data preprocessing

Google Generative AI — Chatbot using Gemini 2.5 Flash

Plotly — Interactive visualizations

Pandas — Data manipulation

🎭 Available Personas

Executive — High time value, productivity-focused

Student — Limited budget, flexible schedule

Parent — Prioritizes predictability and family comfort

Retiree — Values comfort, low stress tolerance

Tourist — Seeks experiences, medium budget

Digital Nomad — Highly flexible, works remotely

Explorer — Adventurous, tolerant of setbacks

VIP — Maximum comfort, willing to pay for guarantees

Immigrant — Essential travel, cost-sensitive

Commuter — Frequent traveler, prioritizes efficiency

📝 Technical Notes

Python 3.13: Fully compatible (google-generativeai installed successfully)

Protobuf: Version 5.29.5 (automatic downgrade from 6.33.4)

sklearn: Version warning (1.6.1 → 1.8.0) is non-blocking

Default Port: http://localhost:8501

Development Mode: Hot reload enabled

🔗 Useful Links

Project Drive: https://drive.google.com/drive/folders/1LoptgYXrfqikYUDppOhGRmC-DARjlwAf

Gemini API: https://ai.google.dev/gemini-api/docs

📄 License

Academic/demo project.
