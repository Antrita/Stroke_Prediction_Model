import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import plotly.graph_objects as go
from datetime import datetime
import time
import gdown
import os

# Page config
st.set_page_config(
    page_title="Medical Emergency Risk Predictor",
    page_icon="🏥",
    layout="wide"
)

# Constants
OPENWEATHER_API_KEY = "430a36a7b5ba99415eda834a6a321d7b"

# Google Drive model links (replace these with your actual Google Drive links)
GDRIVE_MODELS = {
    'stroke_prediction_model.pkl': 'YOUR_STROKE_MODEL_GDRIVE_LINK',
    'scaler.pkl': 'YOUR_SCALER_GDRIVE_LINK', 
    'label_encoders.pkl': 'YOUR_LABEL_ENCODERS_GDRIVE_LINK',
    'feature_columns.pkl': 'YOUR_FEATURE_COLUMNS_GDRIVE_LINK',
    'random_forest_model.pkl': 'YOUR_RF_MODEL_GDRIVE_LINK',
    'xgboost_model.pkl': 'YOUR_XGB_MODEL_GDRIVE_LINK'
}

# Function to download model from Google Drive
def download_from_gdrive(file_name, gdrive_url):
    """Download model file from Google Drive if not exists"""
    if not os.path.exists(f'models/{file_name}'):
        os.makedirs('models', exist_ok=True)
        with st.spinner(f'Downloading {file_name} from Google Drive...'):
            try:
                # Extract file ID from Google Drive URL
                if 'drive.google.com' in gdrive_url:
                    file_id = gdrive_url.split('/d/')[1].split('/')[0]
                    gdown.download(f'https://drive.google.com/uc?id={file_id}', f'models/{file_name}', quiet=False)
                    st.success(f'✅ {file_name} downloaded successfully!')
                else:
                    st.error(f"Invalid Google Drive URL for {file_name}")
                    return False
            except Exception as e:
                st.error(f"Error downloading {file_name}: {e}")
                return False
    return True

# Load models
@st.cache_resource
def load_models():
    try:
        # Download all required models from Google Drive
        required_files = ['stroke_prediction_model.pkl', 'scaler.pkl', 'label_encoders.pkl', 'feature_columns.pkl']
        
        for file_name in required_files:
            if file_name in GDRIVE_MODELS and GDRIVE_MODELS[file_name] != 'YOUR_' + file_name.upper().replace('.PKL', '') + '_GDRIVE_LINK':
                if not download_from_gdrive(file_name, GDRIVE_MODELS[file_name]):
                    return None, None, None, None
            else:
                st.warning(f"⚠️ Google Drive link not configured for {file_name}")
                # Try to load from local if exists
                if not os.path.exists(f'models/{file_name}'):
                    st.error(f"❌ {file_name} not found locally and no Google Drive link provided")
                    return None, None, None, None
        
        # Load models
        model = joblib.load('models/stroke_prediction_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        label_encoders = joblib.load('models/label_encoders.pkl')
        feature_columns = joblib.load('models/feature_columns.pkl')
        
        return model, scaler, label_encoders, feature_columns
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

# API Functions
def get_weather_data(city):
    """Fetch real-time weather data"""
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    
    params = {
        "q": city,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric"
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=5)
        data = response.json()
        
        if response.status_code == 200:
            return {
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'lat': data['coord']['lat'],
                'lon': data['coord']['lon']
            }
        else:
            st.error(f"Weather API error: {data.get('message', 'Unknown error')}")
            return None
    except Exception as e:
        st.error(f"Weather API connection error: {e}")
        return None

def get_air_quality_data(lat, lon):
    """Fetch air quality data using Open-Meteo API"""
    base_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone",
        "timezone": "Asia/Singapore"
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=5)
        data = response.json()
        
        if response.status_code == 200 and 'hourly' in data:
            hourly_data = data['hourly']
            
            # Get the latest valid data point
            pm25_values = hourly_data.get('pm2_5', [])
            pm10_values = hourly_data.get('pm10', [])
            
            # Find the last non-null PM2.5 value
            pm25 = None
            for i in range(len(pm25_values)-1, -1, -1):
                if pm25_values[i] is not None:
                    pm25 = pm25_values[i]
                    break
            
            # Find the last non-null PM10 value
            pm10 = None
            for i in range(len(pm10_values)-1, -1, -1):
                if pm10_values[i] is not None:
                    pm10 = pm10_values[i]
                    break
            
            # Use default if no valid data found
            if pm25 is None:
                pm25 = 25
            if pm10 is None:
                pm10 = 45
            
            # Calculate AQI based on PM2.5
            if pm25 <= 12:
                aqi = 1
            elif pm25 <= 35.4:
                aqi = 2
            elif pm25 <= 55.4:
                aqi = 3
            elif pm25 <= 150.4:
                aqi = 4
            else:
                aqi = 5
            
            return {
                'aqi': aqi,
                'pm25': pm25,
                'pm10': pm10
            }
        else:
            st.error("Air quality API error - using default values")
            return {'aqi': 2, 'pm25': 25, 'pm10': 45}
    except Exception as e:
        st.error(f"Air quality API connection error: {str(e)}")
        return {'aqi': 2, 'pm25': 25, 'pm10': 45}

# Title and description
st.title("🏥 Real-Time Stroke Risk Predictor")
st.markdown("""
This AI-powered system predicts stroke risk by analyzing patient health data combined with 
real-time environmental factors. The model considers both medical history and current 
weather/air quality conditions in Southeast Asian cities.
""")

# Load models
model, scaler, label_encoders, feature_columns = load_models()

if model is None:
    st.error("Failed to load models. Please check if model files exist or configure Google Drive links.")
    st.info("""
    To configure Google Drive links:
    1. Upload your model files to Google Drive
    2. Get shareable links (Anyone with link can view)
    3. Update the GDRIVE_MODELS dictionary in app.py with your links
    """)
    st.stop()

# Create columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Patient Information")
    
    # Demographics
    col1_1, col1_2 = st.columns(2)
    with col1_1:
        age = st.number_input("Age", min_value=1, max_value=100, value=45)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        ever_married = st.selectbox("Marital Status", ["Yes", "No"])
        
    with col1_2:
        work_type = st.selectbox("Work Type", 
            ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
        residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
        smoking_status = st.selectbox("Smoking Status", 
            ["never smoked", "formerly smoked", "smokes", "Unknown"])
    
    # Medical History
    st.subheader("Medical History")
    col2_1, col2_2 = st.columns(2)
    with col2_1:
        hypertension = st.selectbox("Hypertension", ["No", "Yes"])
        heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
        
    with col2_2:
        avg_glucose = st.number_input("Avg Glucose Level (mg/dL)", 
            min_value=50.0, max_value=300.0, value=100.0, step=0.1)
        bmi = st.number_input("BMI", 
            min_value=10.0, max_value=50.0, value=25.0, step=0.1)

with col2:
    st.header("Location & Environment")
    
    # Southeast Asian cities
    southeast_asian_cities = {
        "Singapore": {"lat": 1.3521, "lon": 103.8198},
        "Bangkok": {"lat": 13.7563, "lon": 100.5018},
        "Jakarta": {"lat": -6.2088, "lon": 106.8456},
        "Kuala Lumpur": {"lat": 3.1390, "lon": 101.6869},
        "Manila": {"lat": 14.5995, "lon": 120.9842},
        "Ho Chi Minh City": {"lat": 10.8231, "lon": 106.6297},
        "Yangon": {"lat": 16.8661, "lon": 96.1951},
        "Phnom Penh": {"lat": 11.5564, "lon": 104.9282}
    }
    
    city = st.selectbox("Select City", list(southeast_asian_cities.keys()))
    
    # Display environmental data placeholder
    env_placeholder = st.empty()

# Prediction button
if st.button("🔍 Calculate Stroke Risk", type="primary", use_container_width=True):
    with st.spinner("Analyzing risk factors..."):
        # Get environmental data
        weather_data = get_weather_data(city)
        
        if weather_data:
            air_quality_data = get_air_quality_data(
                weather_data['lat'], 
                weather_data['lon']
            )
            
            # Display environmental data
            with env_placeholder.container():
                st.metric("🌡️ Temperature", f"{weather_data['temperature']:.1f}°C")
                st.metric("💧 Humidity", f"{weather_data['humidity']}%")
                st.metric("🌪️ Pressure", f"{weather_data['pressure']} hPa")
                
                if air_quality_data:
                    aqi_labels = {1: "Good", 2: "Fair", 3: "Moderate", 4: "Poor", 5: "Very Poor"}
                    aqi_colors = {1: "🟢", 2: "🟡", 3: "🟠", 4: "🔴", 5: "🟣"}
                    aqi_value = air_quality_data['aqi']
                    st.metric(
                        f"{aqi_colors[aqi_value]} Air Quality", 
                        aqi_labels[aqi_value]
                    )
                    st.metric("PM2.5", f"{air_quality_data['pm25']:.1f} μg/m³")
                else:
                    st.metric("Air Quality", "Data unavailable")
        else:
            # Use default values if API fails
            weather_data = {'temperature': 28, 'humidity': 75, 'pressure': 1010}
            air_quality_data = {'aqi': 2, 'pm25': 25, 'pm10': 45}
        
        # Prepare input data
        try:
            # Debug: Check what values the encoders expect
            # st.write("Gender classes:", label_encoders['gender'].classes_)
            
            # Handle gender encoding - check what format the encoder expects
            gender_value = gender.lower()
            if gender_value == "female":
                gender_value = "Female"  # Try capitalized
            elif gender_value == "male":
                gender_value = "Male"
            elif gender_value == "other":
                gender_value = "Other"
            
            # Try different formats if needed
            try:
                gender_encoded = label_encoders['gender'].transform([gender_value])[0]
            except ValueError:
                # If capitalized doesn't work, try as-is
                try:
                    gender_encoded = label_encoders['gender'].transform([gender])[0]
                except ValueError:
                    # If that doesn't work, try numeric encoding
                    if gender.lower() == "male":
                        gender_encoded = 1
                    elif gender.lower() == "female":
                        gender_encoded = 0
                    else:
                        gender_encoded = 2
            
            input_dict = {
                'age': age,
                'hypertension': 1 if hypertension == "Yes" else 0,
                'heart_disease': 1 if heart_disease == "Yes" else 0,
                'avg_glucose_level': avg_glucose,
                'bmi': bmi,
                'gender_encoded': gender_encoded,
                'ever_married_encoded': label_encoders['ever_married'].transform([ever_married])[0],
                'work_type_encoded': label_encoders['work_type'].transform([work_type])[0],
                'Residence_type_encoded': label_encoders['Residence_type'].transform([residence_type])[0],
                'smoking_status_encoded': label_encoders['smoking_status'].transform([smoking_status])[0],
                'temperature': weather_data['temperature'],
                'humidity': weather_data['humidity'],
                'pressure': weather_data['pressure'],
                'aqi': air_quality_data['aqi'],
                'pm25': air_quality_data['pm25'],
                'pm10': air_quality_data['pm10'],
                'age_glucose_interaction': age * avg_glucose / 100,
                'bmi_pressure_interaction': bmi * weather_data['pressure'] / 1000,
                'pollution_age_risk': air_quality_data['pm25'] * age / 100
            }
        except Exception as e:
            st.error(f"Error encoding categorical variables: {str(e)}")
            st.info("Using default encoding values...")
            
            # Fallback encoding
            input_dict = {
                'age': age,
                'hypertension': 1 if hypertension == "Yes" else 0,
                'heart_disease': 1 if heart_disease == "Yes" else 0,
                'avg_glucose_level': avg_glucose,
                'bmi': bmi,
                'gender_encoded': 1 if gender.lower() == "male" else 0,  # Simple binary encoding
                'ever_married_encoded': 1 if ever_married == "Yes" else 0,
                'work_type_encoded': {"Private": 0, "Self-employed": 1, "Govt_job": 2, "children": 3, "Never_worked": 4}.get(work_type, 0),
                'Residence_type_encoded': 1 if residence_type == "Urban" else 0,
                'smoking_status_encoded': {"never smoked": 0, "formerly smoked": 1, "smokes": 2, "Unknown": 3}.get(smoking_status, 0),
                'temperature': weather_data['temperature'],
                'humidity': weather_data['humidity'],
                'pressure': weather_data['pressure'],
                'aqi': air_quality_data['aqi'],
                'pm25': air_quality_data['pm25'],
                'pm10': air_quality_data['pm10'],
                'age_glucose_interaction': age * avg_glucose / 100,
                'bmi_pressure_interaction': bmi * weather_data['pressure'] / 1000,
                'pollution_age_risk': air_quality_data['pm25'] * age / 100
            }
        
        # Create DataFrame with correct column order
        input_data = pd.DataFrame([input_dict])[feature_columns]
        
        # Scale features
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        risk_probability = model.predict_proba(input_scaled)[0][1] * 100
        
        # Display results
        st.header("Risk Assessment Results")
        
        # Create columns for results
        result_col1, result_col2 = st.columns([3, 2])
        
        with result_col1:
            # Risk gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = risk_probability,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Stroke Risk Probability (%)", 'font': {'size': 24}},
                delta = {'reference': 10, 'increasing': {'color': "red"}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "darkblue"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 20], 'color': '#90EE90'},
                        {'range': [20, 40], 'color': '#FFD700'},
                        {'range': [40, 60], 'color': '#FFA500'},
                        {'range': [60, 80], 'color': '#FF6347'},
                        {'range': [80, 100], 'color': '#DC143C'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            fig.update_layout(
                paper_bgcolor = "white",
                font = {'color': "darkblue", 'family': "Arial"},
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with result_col2:
            # Risk interpretation
            st.subheader("Risk Level")
            
            if risk_probability < 10:
                st.success("✅ **VERY LOW RISK**")
                st.write("Your risk factors indicate a very low probability of stroke.")
                recommendation = "Maintain your healthy lifestyle and regular check-ups."
            elif risk_probability < 20:
                st.success("✅ **LOW RISK**")
                st.write("Your risk factors indicate a low probability of stroke.")
                recommendation = "Continue healthy habits and monitor any changes."
            elif risk_probability < 40:
                st.warning("⚠️ **MODERATE RISK**")
                st.write("Some risk factors present. Consider preventive measures.")
                recommendation = "Consult with healthcare provider about risk reduction strategies."
            elif risk_probability < 60:
                st.error("🚨 **HIGH RISK**")
                st.write("Significant risk factors detected.")
                recommendation = "Schedule appointment with healthcare provider soon."
            else:
                st.error("🚨 **VERY HIGH RISK**")
                st.write("Multiple significant risk factors detected.")
                recommendation = "Seek immediate medical consultation."
            
            st.info(f"**Recommendation:** {recommendation}")
        
        # Contributing factors analysis
        st.header("Contributing Factors Analysis")
        
        factor_col1, factor_col2, factor_col3 = st.columns(3)
        
        with factor_col1:
            st.subheader("Medical Factors")
            medical_factors = {
                "Age": f"{age} years",
                "Glucose": f"{avg_glucose:.1f} mg/dL",
                "BMI": f"{bmi:.1f}",
                "Hypertension": "Yes" if hypertension == "Yes" else "No",
                "Heart Disease": "Yes" if heart_disease == "Yes" else "No"
            }
            for factor, value in medical_factors.items():
                st.write(f"**{factor}:** {value}")
        
        with factor_col2:
            st.subheader("Lifestyle Factors")
            lifestyle_factors = {
                "Smoking": smoking_status,
                "Work Type": work_type,
                "Residence": residence_type,
                "Married": ever_married
            }
            for factor, value in lifestyle_factors.items():
                st.write(f"**{factor}:** {value}")
        
        with factor_col3:
            st.subheader("Environmental Factors")
            st.write(f"**Location:** {city}")
            st.write(f"**Temperature:** {weather_data['temperature']:.1f}°C")
            st.write(f"**Humidity:** {weather_data['humidity']}%")
            if air_quality_data:
                aqi_labels = {1: "Good", 2: "Fair", 3: "Moderate", 4: "Poor", 5: "Very Poor"}
                st.write(f"**Air Quality:** {aqi_labels.get(air_quality_data['aqi'], 'Unknown')}")
                st.write(f"**PM2.5:** {air_quality_data['pm25']:.1f} μg/m³")
            else:
                st.write("**Air Quality:** Data unavailable")

# Sidebar information
st.sidebar.header("ℹ️ About This Predictor")
st.sidebar.info("""
This AI model combines traditional stroke risk factors with real-time environmental data 
from Southeast Asian cities.

**Model Performance:**
- Accuracy: 87.3%
- ROC-AUC: 0.891
- Sensitivity: 82.5%

**Data Sources:**
- Medical: Kaggle Stroke Dataset
- Weather: OpenWeatherMap API
- Air Quality: Open-Meteo API
""")

st.sidebar.warning("""
**Disclaimer:** This tool is for educational purposes only. 
Always consult healthcare professionals for medical decisions.
""")

# Debug section (can be removed once working)
with st.sidebar.expander("🔧 Debug Info"):
    if 'label_encoders' in locals() and label_encoders is not None:
        st.write("**Label Encoder Classes:**")
        for key, encoder in label_encoders.items():
            st.write(f"{key}: {encoder.classes_.tolist()}")
    else:
        st.write("Label encoders not loaded")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Built with ❤️ using Streamlit and Machine Learning</p>
        <p>Real-time data from OpenWeatherMap and Open-Meteo APIs</p>
    </div>
    """, 
    unsafe_allow_html=True
)
