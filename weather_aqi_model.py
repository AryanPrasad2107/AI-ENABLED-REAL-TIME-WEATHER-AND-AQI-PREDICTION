import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor,ExtraTreesRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import pytz

API_KEY = '3eb116bbba522361fb8baf973094dd85'
BASE_URL = 'https://api.openweathermap.org/data/2.5/'

# AQI category label 

AQI_CATEGORY = {
    1: "Good",
    2: "Fair",
    3: "Moderate",
    4: "Poor",
    5: "Very Poor"
}


# Fetch current weather

def get_current_weather(city):
    url = f"{BASE_URL}weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()
    return {
        'city':           data['name'],
        'current_temp':   round(data['main']['temp']),
        'feels_like':     round(data['main']['feels_like']),
        'temp_min':       round(data['main']['temp_min']),
        'temp_max':       round(data['main']['temp_max']),
        'humidity':       round(data['main']['humidity']),
        'description':    data['weather'][0]['description'],
        'country':        data['sys']['country'],
        'wind_gust_dir':  data['wind']['deg'],
        'pressure':       data['main']['pressure'],
        'Wind_Gust_Speed': data['wind']['speed'],
        # Coordinates needed for Air Pollution API
        'lat':            data['coord']['lat'],
        'lon':            data['coord']['lon'],
    }

# Fetch live AQI 

def get_current_aqi(lat, lon):
    """
    Returns live AQI index (1–5) and pollutant
    component concentrations for the given coordinates.
    OpenWeatherMap docs:
      https://openweathermap.org/api/air-pollution
    """
    url = (
        f"http://api.openweathermap.org/data/2.5/air_pollution"
        f"?lat={lat}&lon={lon}&appid={API_KEY}"
    )
    response = requests.get(url)
    data = response.json()

    aqi_index  = data['list'][0]['main']['aqi']          # 1–5
    components = data['list'][0]['components']           # pollutant µg/m³

    return {
        'aqi_index': aqi_index,
        'aqi_label': AQI_CATEGORY.get(aqi_index, 'Unknown'),
        'co':    components.get('co',   0),
        'no2':   components.get('no2',  0),
        'o3':    components.get('o3',   0),
        'so2':   components.get('so2',  0),
        'pm2_5': components.get('pm2_5', 0),
        'pm10':  components.get('pm10', 0),
    }

#Load & preprocess AQI historical dataset

def read_aqi_data(filename):
    """
    Reads the global AQI CSV.
    Columns used: PM2.5, PM10, NO2, SO2, CO, O3, AQI
    """
    df = pd.read_csv(filename)
    df = df.dropna()
    df = df.drop_duplicates()
    return df

# Feature engineering helper for AQI model

def _build_aqi_features(pm25, pm10, no2, so2, co, o3):
    """
    Expands 6 raw pollutant readings into 14 features:
      • Original 6 values
      • PM2.5 × PM10  (particle interaction)
      • PM2.5 / PM10  (fine-to-coarse ratio)
      • NO2 + SO2     (combined gas burden)
      • CO × O3       (oxidant interaction)
      • PM2.5²        (non-linear particle effect)
      • PM10²         (non-linear particle effect)
      • CO²           (non-linear CO effect)
      • NO2 × O3      (photochemical smog proxy)
    Richer features compensate for the small dataset size
    and the weak linear correlations between pollutants
    and AQI.
    """
    base = np.array([pm25, pm10, no2, so2, co, o3], dtype=float)
    interactions = np.array([
        pm25 * pm10,
        pm25 / (pm10 + 1e-9),
        no2  + so2,
        co   * o3,
        pm25 ** 2,
        pm10 ** 2,
        co   ** 2,
        no2  * o3,
    ], dtype=float)
    return np.concatenate([base, interactions])


def _aqi_feature_matrix(df):
    """Builds the engineered feature matrix from an AQI dataframe."""
    rows = [_build_aqi_features(
        r['PM2.5'], r['PM10'], r['NO2'], r['SO2'], r['CO'], r['O3']
    ) for _, r in df.iterrows()]
    return np.array(rows)


#AQI model

def train_aqi_model(aqi_df):
    X = _aqi_feature_matrix(aqi_df)
    y = aqi_df['AQI'].values

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Grid-search
    param_grid = {
        'n_estimators': [200, 300, 500],
        'max_depth':    [3, 4, 5],
        'min_samples_leaf': [1, 2, 3],
        'max_features': ['sqrt', 'log2', 0.7],
    }
    gs = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid, cv=kf,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    gs.fit(X, y)

    #Stacking ensemble 
    estimators = [
        ('rf',  RandomForestRegressor(**gs.best_params_, random_state=42)),
        ('et',  ExtraTreesRegressor(n_estimators=300, max_depth=5, random_state=42)),
        ('svr', Pipeline([
            ('sc', StandardScaler()),
            ('m',  SVR(C=100, gamma='scale', kernel='rbf'))
        ])),
    ]
    model = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=1.0),
        cv=5
    )

    #Cross-validated MSE
    cv_scores = cross_val_score(
        model, X, y, cv=kf, scoring='neg_mean_squared_error'
    )
    cv_mse = -cv_scores.mean()
    print(f"Mean Squared Error for AQI Model (5-fold CV): {round(cv_mse, 2)}"
          f"  ±  {round(cv_scores.std(), 2)}")

    model.fit(X, y)
    return model


#Predict AQI from live pollutant readings

def predict_aqi_from_components(aqi_model, live_aqi_data):
    """
    Applies the same feature engineering used during
    training before passing live OWM readings to the model.
    """
    feat = _build_aqi_features(
        live_aqi_data['pm2_5'], live_aqi_data['pm10'],
        live_aqi_data['no2'],   live_aqi_data['so2'],
        live_aqi_data['co'],    live_aqi_data['o3'],
    ).reshape(1, -1)
    predicted_aqi = aqi_model.predict(feat)[0]
    return round(predicted_aqi, 1)

# Predict future AQI for next N hours

def predict_future_aqi(aqi_df, current_aqi_value, steps=5):
    """
    Trains a simple lag-1 RandomForestRegressor on
    historical AQI values from the CSV and projects
    future AQI for the next `steps` hours.
    """
    aqi_series = aqi_df['AQI'].values
    X = aqi_series[:-1].reshape(-1, 1)
    y = aqi_series[1:]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    predictions = [current_aqi_value]
    for _ in range(steps):
        next_val = model.predict(np.array([[predictions[-1]]]))[0]
        predictions.append(next_val)

    return [round(v, 1) for v in predictions[1:]]


def read_historical_data(filename):
    df = pd.read_csv(filename)
    df = df.dropna()
    df = df.drop_duplicates()
    return df

def prepare_data(data):
    df = data.copy()
    le = LabelEncoder()
    df['WindGustDir']   = le.fit_transform(df['WindGustDir'])
    df['RainTomorrow']  = le.fit_transform(df['RainTomorrow'])
    X = df[['MinTemp', 'MaxTemp', 'WindGustDir',
            'WindGustSpeed', 'Humidity', 'Pressure', 'Temp']]
    y = df['RainTomorrow']
    return X, y, le

def train_rain_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Mean Squared Error for Rain Model:",
          round(mean_squared_error(y_test, y_pred), 4))
    return model

def prepare_regression_data(data, feature):
    X, y = [], []
    for i in range(len(data) - 1):
        X.append(data[feature].iloc[i])
        y.append(data[feature].iloc[i + 1])
    return np.array(X).reshape(-1, 1), np.array(y)

def train_regression_model(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def predict_future(model, current_value, steps=5):
    predictions = [current_value]
    for _ in range(steps):
        next_val = model.predict(np.array([[predictions[-1]]]))[0]
        predictions.append(next_val)
    return predictions[1:]

# AQI numeric

def aqi_to_category(aqi_value):
    if aqi_value <= 50:   return "Good"
    if aqi_value <= 100:  return "Moderate"
    if aqi_value <= 150:  return "Unhealthy for Sensitive Groups"
    if aqi_value <= 200:  return "Unhealthy"
    if aqi_value <= 300:  return "Very Unhealthy"
    return "Hazardous"

# Main 

def weather_view():
    city = input("Enter the city name: ")

    #Current weather 
    current_weather = get_current_weather(city)

    #Live AQI from OpenWeatherMap 
    print("\nFetching live AQI data...")
    live_aqi = get_current_aqi(current_weather['lat'], current_weather['lon'])

    #Load datasets 
    historical_data = read_historical_data(
        'C:\\Users\\aryan\\OneDrive\\Desktop\\Microsoft Internship\\Project\\weather.csv'
    )
    aqi_data = read_aqi_data(
        'C:\\Users\\aryan\\OneDrive\\Desktop\\Microsoft Internship\\Project\\global_aqi.csv'
    )

    #Train all models 
    X, y, le = prepare_data(historical_data)
    rain_model = train_rain_model(X, y)
    aqi_model  = train_aqi_model(aqi_data)

    # Wind direction encoding 
    wind_deg = current_weather['wind_gust_dir'] % 360
    compass_points = [
        ("N",   0,     11.25), ("NNE", 11.25,  33.75),
        ("NE",  33.75, 56.25), ("ENE", 56.25,  78.75),
        ("E",   78.75, 101.25),("ESE", 101.25, 123.75),
        ("SE",  123.75,146.25),("SSE", 146.25, 168.75),
        ("S",   168.75,191.25),("SSW", 191.25, 213.75),
        ("SW",  213.75,236.25),("WSW", 236.25, 258.75),
        ("W",   258.75,281.25),("WNW", 281.25, 303.75),
        ("NW",  303.75,326.25),("NNW", 326.25, 348.75),
    ]
    compass_direction = next(
        pt for pt, start, end in compass_points if start <= wind_deg < end
    )
    compass_encoded = (
        le.transform([compass_direction])[0]
        if compass_direction in le.classes_ else -1
    )

    # Rain prediction 
    current_df = pd.DataFrame([{
        'MinTemp':      current_weather['temp_min'],
        'MaxTemp':      current_weather['temp_max'],
        'WindGustDir':  compass_encoded,
        'WindGustSpeed':current_weather['Wind_Gust_Speed'],
        'Humidity':     current_weather['humidity'],
        'Pressure':     current_weather['pressure'],
        'Temp':         current_weather['current_temp'],
    }])
    rain_prediction = rain_model.predict(current_df)[0]

    # AQI prediction from live components 
    predicted_aqi_value = predict_aqi_from_components(aqi_model, live_aqi)

    # Future temp / humidity / AQI 
    X_temp, y_temp = prepare_regression_data(historical_data, 'Temp')
    X_hum,  y_hum  = prepare_regression_data(historical_data, 'Humidity')
    temp_model = train_regression_model(X_temp, y_temp)
    hum_model  = train_regression_model(X_hum,  y_hum)

    future_temp     = predict_future(temp_model, current_weather['temp_min'])
    future_humidity = predict_future(hum_model,  current_weather['humidity'])
    future_aqi      = predict_future_aqi(aqi_data, predicted_aqi_value, steps=5)

    # Time labels 
    timezone     = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(timezone)
    next_hour    = (current_time + timedelta(hours=1)).replace(
                      minute=0, second=0, microsecond=0)
    future_times = [
        (next_hour + timedelta(hours=i)).strftime("%H:00") for i in range(5)
    ]

    # Print results 
    print("\n" + "═" * 45)
    print(f"{current_weather['city']}, {current_weather['country']}")
    print("═" * 45)

    print("\n Weather")
    print(f"  Current Temperature : {current_weather['current_temp']}°C")
    print(f"  Feels Like          : {current_weather['feels_like']}°C")
    print(f"  Min / Max           : {current_weather['temp_min']}°C / {current_weather['temp_max']}°C")
    print(f"  Humidity            : {current_weather['humidity']}%")
    print(f"  Condition           : {current_weather['description'].title()}")
    print(f"  Rain Tomorrow       : {'Yes 🌧' if rain_prediction else 'No ☀️'}")

    print("\nAir Quality Index (AQI)")
    print(f"  Live AQI Index      : {live_aqi['aqi_index']} / 5  ({live_aqi['aqi_label']})")
    print(f"  Predicted AQI Score : {predicted_aqi_value}  ({aqi_to_category(predicted_aqi_value)})")
    print(f"  PM2.5               : {live_aqi['pm2_5']} µg/m³")
    print(f"  PM10                : {live_aqi['pm10']} µg/m³")
    print(f"  NO₂                 : {live_aqi['no2']} µg/m³")
    print(f"  SO₂                 : {live_aqi['so2']} µg/m³")
    print(f"  CO                  : {live_aqi['co']} µg/m³")
    print(f"  O₃                  : {live_aqi['o3']} µg/m³")

    print("\nHourly Forecast")
    print(f"  {'Time':<8} {'Temp (°C)':<14} {'Humidity (%)':<16} {'AQI':<10} Category")
    print(f"  {'─'*7} {'─'*13} {'─'*15} {'─'*9} {'─'*30}")
    for time, temp, hum, aqi_val in zip(
            future_times, future_temp, future_humidity, future_aqi):
        print(
            f"  {time:<8} {round(temp,1):<14} {round(hum,1):<16}"
            f" {aqi_val:<10} {aqi_to_category(aqi_val)}"
        )

    print("\n" + "═" * 45)


if __name__ == '__main__':
    weather_view()
