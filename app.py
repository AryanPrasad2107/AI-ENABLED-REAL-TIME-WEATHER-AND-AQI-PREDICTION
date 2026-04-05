import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import pytz

#CONFIGURATION 
API_KEY = '3eb116bbba522361fb8baf973094dd85'
BASE_URL = 'https://api.openweathermap.org/data/2.5/'


WEATHER_CSV_PATH = r'C:\Users\aryan\OneDrive\Desktop\Microsoft Internship\Project\weather.csv'
AQI_CSV_PATH = r'C:\Users\aryan\OneDrive\Desktop\Microsoft Internship\Project\global_aqi.csv'

AQI_CATEGORY = {1: "Good", 2: "Fair", 3: "Moderate", 4: "Poor", 5: "Very Poor"}

def get_current_weather(city):
    url = f"{BASE_URL}weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code != 200:
        return None
    data = response.json()
    return {
        'city': data['name'], 'current_temp': round(data['main']['temp']),
        'feels_like': round(data['main']['feels_like']), 'temp_min': round(data['main']['temp_min']),
        'temp_max': round(data['main']['temp_max']), 'humidity': round(data['main']['humidity']),
        'description': data['weather'][0]['description'], 'country': data['sys']['country'],
        'wind_gust_dir': data['wind']['deg'], 'pressure': data['main']['pressure'],
        'Wind_Gust_Speed': data['wind']['speed'], 'lat': data['coord']['lat'], 'lon': data['coord']['lon'],
    }

def get_current_aqi(lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
    response = requests.get(url)
    data = response.json()
    aqi_index = data['list'][0]['main']['aqi']
    components = data['list'][0]['components']
    return {
        'aqi_index': aqi_index, 'aqi_label': AQI_CATEGORY.get(aqi_index, 'Unknown'),
        'co': components.get('co', 0), 'no2': components.get('no2', 0),
        'o3': components.get('o3', 0), 'so2': components.get('so2', 0),
        'pm2_5': components.get('pm2_5', 0), 'pm10': components.get('pm10', 0),
    }

def _build_aqi_features(pm25, pm10, no2, so2, co, o3):
    base = np.array([pm25, pm10, no2, so2, co, o3], dtype=float)
    interactions = np.array([
        pm25 * pm10, pm25 / (pm10 + 1e-9), no2 + so2, co * o3,
        pm25 ** 2, pm10 ** 2, co ** 2, no2 * o3,
    ], dtype=float)
    return np.concatenate([base, interactions])

def _aqi_feature_matrix(df):
    rows = [_build_aqi_features(r['PM2.5'], r['PM10'], r['NO2'], r['SO2'], r['CO'], r['O3']) for _, r in df.iterrows()]
    return np.array(rows)

def aqi_to_category(aqi_value):
    if aqi_value <= 50: return "Good"
    if aqi_value <= 100: return "Moderate"
    if aqi_value <= 150: return "Unhealthy for Sensitive Groups"
    if aqi_value <= 200: return "Unhealthy"
    if aqi_value <= 300: return "Very Unhealthy"
    return "Hazardous"

# ML TRAINING FUNCTIONS 
def load_and_train_models():
    # Load Data
    historical_data = pd.read_csv(WEATHER_CSV_PATH).dropna().drop_duplicates()
    aqi_data = pd.read_csv(AQI_CSV_PATH).dropna().drop_duplicates()

    # Train Rain Model
    le = LabelEncoder()
    historical_data['WindGustDir_Encoded'] = le.fit_transform(historical_data['WindGustDir'])
    historical_data['RainTomorrow_Encoded'] = LabelEncoder().fit_transform(historical_data['RainTomorrow'])
    X_rain = historical_data[['MinTemp', 'MaxTemp', 'WindGustDir_Encoded', 'WindGustSpeed', 'Humidity', 'Pressure', 'Temp']]
    y_rain = historical_data['RainTomorrow_Encoded']
    rain_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rain_model.fit(X_rain, y_rain)

    # Train AQI Model
    X_aqi = _aqi_feature_matrix(aqi_data)
    y_aqi = aqi_data['AQI'].values
    estimators = [
        ('rf', RandomForestRegressor(max_depth=5, min_samples_leaf=1, n_estimators=500, random_state=42)),
        ('et', ExtraTreesRegressor(n_estimators=300, max_depth=5, random_state=42)),
        ('svr', Pipeline([('sc', StandardScaler()), ('m', SVR(C=100, gamma='scale', kernel='rbf'))])),
    ]
    aqi_model = StackingRegressor(estimators=estimators, final_estimator=Ridge(alpha=1.0), cv=5)
    aqi_model.fit(X_aqi, y_aqi)

    # Train Future Temp & Hum Models
    def prep_reg(data, feature):
        X, y = [], []
        for i in range(len(data) - 1):
            X.append(data[feature].iloc[i])
            y.append(data[feature].iloc[i + 1])
        return np.array(X).reshape(-1, 1), np.array(y)

    X_temp, y_temp = prep_reg(historical_data, 'Temp')
    X_hum, y_hum = prep_reg(historical_data, 'Humidity')
    temp_model = RandomForestRegressor(n_estimators=10, random_state=42).fit(X_temp, y_temp)
    hum_model = RandomForestRegressor(n_estimators=10, random_state=42).fit(X_hum, y_hum)

    return rain_model, aqi_model, temp_model, hum_model, aqi_data, le

# STREAMLIT GUI LAYOUT
st.set_page_config(page_title="Weather & AQI Predictor", page_icon="🌤️", layout="wide")

st.title("🌤️ AI Weather & Air Quality Forecaster")
st.markdown("Powered by OpenWeatherMap and Machine Learning")

# Load models implicitly using cache
try:
    rain_model, aqi_model, temp_model, hum_model, aqi_data, le = load_and_train_models()
except FileNotFoundError:
    st.error("Error: Could not find the dataset CSV files. Please check the file paths at the top of the script.")
    st.stop()

# Search Bar
col1, col2 = st.columns([3, 1])
with col1:
    city = st.text_input("Enter City Name", "London")
with col2:
    st.write("") # Spacer
    st.write("") # Spacer
    search_clicked = st.button("Get Forecast", use_container_width=True)

if search_clicked and city:
    with st.spinner("Fetching live data and running predictions..."):
        weather = get_current_weather(city)
        
        if not weather:
            st.error("City not found. Please try again.")
        else:
            live_aqi = get_current_aqi(weather['lat'], weather['lon'])

            # --- ML PREDICTIONS ---
            # 1. Rain Prediction
            wind_deg = weather['wind_gust_dir'] % 360
            compass_points = [("N", 0, 11.25), ("NNE", 11.25, 33.75), ("NE", 33.75, 56.25), ("ENE", 56.25, 78.75), ("E", 78.75, 101.25), ("ESE", 101.25, 123.75), ("SE", 123.75, 146.25), ("SSE", 146.25, 168.75), ("S", 168.75, 191.25), ("SSW", 191.25, 213.75), ("SW", 213.75, 236.25), ("WSW", 236.25, 258.75), ("W", 258.75, 281.25), ("WNW", 281.25, 303.75), ("NW", 303.75, 326.25), ("NNW", 326.25, 348.75)]
            compass_dir = next(pt for pt, start, end in compass_points if start <= wind_deg < end)
            compass_encoded = le.transform([compass_dir])[0] if compass_dir in le.classes_ else -1
            
            rain_df = pd.DataFrame([{
                'MinTemp': weather['temp_min'], 'MaxTemp': weather['temp_max'],
                'WindGustDir_Encoded': compass_encoded, 'WindGustSpeed': weather['Wind_Gust_Speed'],
                'Humidity': weather['humidity'], 'Pressure': weather['pressure'], 'Temp': weather['current_temp']
            }])
            rain_pred = "Yes 🌧️" if rain_model.predict(rain_df)[0] else "No ☀️"

            # 2. Predicted AQI
            feat = _build_aqi_features(live_aqi['pm2_5'], live_aqi['pm10'], live_aqi['no2'], live_aqi['so2'], live_aqi['co'], live_aqi['o3']).reshape(1, -1)
            predicted_aqi_val = round(aqi_model.predict(feat)[0], 1)

            # 3. Future Predictions (Next 5 Hours)
            def predict_future(model, current_value, steps=5):
                preds = [current_value]
                for _ in range(steps):
                    preds.append(model.predict(np.array([[preds[-1]]]))[0])
                return preds[1:]
            
            def predict_future_aqi(aqi_df, current_aqi, steps=5):
                aqi_series = aqi_df['AQI'].values
                X, y = aqi_series[:-1].reshape(-1, 1), aqi_series[1:]
                m = RandomForestRegressor(n_estimators=10, random_state=42).fit(X, y)
                preds = [current_aqi]
                for _ in range(steps): preds.append(m.predict(np.array([[preds[-1]]]))[0])
                return [round(v, 1) for v in preds[1:]]

            future_temp = predict_future(temp_model, weather['temp_min'])
            future_hum = predict_future(hum_model, weather['humidity'])
            future_aqi = predict_future_aqi(aqi_data, predicted_aqi_val)
            
            current_time = datetime.now(pytz.timezone('Asia/Kolkata'))
            future_times = [(current_time + timedelta(hours=i+1)).strftime("%H:00") for i in range(5)]

            # DISPLAY DASHBOARD 
            st.divider()
            st.subheader(f" {weather['city']}, {weather['country']}")
            
            # Weather Metrics
            w_col1, w_col2, w_col3, w_col4 = st.columns(4)
            w_col1.metric("Temperature", f"{weather['current_temp']}°C", f"Feels like {weather['feels_like']}°C")
            w_col2.metric("Condition", weather['description'].title())
            w_col3.metric("Humidity", f"{weather['humidity']}%")
            w_col4.metric("Rain Tomorrow?", rain_pred)

            st.divider()

            # AQI Metrics
            st.subheader(" Air Quality")
            a_col1, a_col2, a_col3 = st.columns(3)
            a_col1.metric("Live OWM AQI Index", f"{live_aqi['aqi_index']} / 5", live_aqi['aqi_label'])
            a_col2.metric("ML Predicted AQI Score", f"{predicted_aqi_val}", aqi_to_category(predicted_aqi_val))
            a_col3.info(f"**Pollutants (µg/m³):** \nPM2.5: {live_aqi['pm2_5']} | PM10: {live_aqi['pm10']}  \nNO₂: {live_aqi['no2']} | SO₂: {live_aqi['so2']}")

            st.divider()

            # Hourly Forecast
            st.subheader(" 5-Hour AI Forecast")
            forecast_df = pd.DataFrame({
                "Time": future_times,
                "Temp (°C)": [round(t, 1) for t in future_temp],
                "Humidity (%)": [round(h, 1) for h in future_hum],
                "Predicted AQI": future_aqi,
                "AQI Category": [aqi_to_category(a) for a in future_aqi]
            })
            st.dataframe(forecast_df, use_container_width=True, hide_index=True)