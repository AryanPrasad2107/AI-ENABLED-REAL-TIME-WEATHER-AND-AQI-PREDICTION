# AI-ENABLED-REAL-TIME-WEATHER-AND-AQI-PREDICTION
# 🌤️ AI Weather & AQI Prediction System

An end-to-end Machine Learning project that predicts **Temperature trends, Rainfall, and Air Quality Index (AQI)** using a combination of **real-time API data and historical datasets**.

---

## 🚀 Project Overview

This project addresses climate variability by building an intelligent system capable of:

- 🌡️ Predicting temperature trends  
- 🌧️ Forecasting rainfall (Yes/No)  
- 🌫️ Estimating Air Quality Index (AQI)  
- 📊 Providing short-term future predictions (next 5 hours)

The system integrates **Machine Learning models + OpenWeatherMap API + Streamlit UI** for real-time insights.

---

## 🧠 Features

- Real-time weather & AQI data using API  
- Machine Learning-based predictions  
- Ensemble model (Stacking) for improved AQI accuracy  
- Hourly future forecasting (Temperature, Humidity, AQI)  
- Interactive and clean Streamlit dashboard  

---

## 🏗️ Tech Stack

- **Programming Language:** Python  
- **Frontend/UI:** Streamlit  
- **Machine Learning:** Scikit-learn  
- **Data Handling:** Pandas, NumPy  
- **API:** OpenWeatherMap API  

---

## 📂 Project Structure
├── app.py # Streamlit UI + model integration
├── weather_aqi_model.py # Core ML models and logic
├── weather.csv # Historical weather dataset
├── global_aqi.csv # AQI dataset
├── requirements.txt # Dependencies
└── README.md # Project documentation

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
git clone https://github.com/your-username/weather-aqi-ml.git
cd weather-aqi-ml

Install Dependencies
pip install -r requirements.txt

Add API Key
API_KEY = "your_openweathermap_api_key"

Run the Project
streamlit run app.py
http://localhost:8501
