import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
import pickle
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta

# --- Setup Upload Directory ---
UPLOAD_FOLDER = "uploads/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Paths ---
CLEAN_BASELINE_FILE = "data/cleaned/water_data_cleaned.csv"
MODEL_FILE = "models/forecast_model.pkl"

# --- Streamlit Page Config ---
st.set_page_config(page_title="El Paso Water Forecast Dashboard", layout="wide")

# --- Custom Hide Footer ---
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# --- Weather Pull ---
def get_live_weather():
    try:
        url = (
            "https://api.open-meteo.com/v1/forecast"
            "?latitude=31.7619"
            "&longitude=-106.4850"
            "&hourly=temperature_2m,relative_humidity_2m,precipitation"
            "&timezone=America/Denver"
        )
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            hourly = data['hourly']
            now = datetime.now(timezone.utc).astimezone()
            current_time_str = now.strftime("%Y-%m-%dT%H:00")
            if current_time_str in hourly['time']:
                idx = hourly['time'].index(current_time_str)
                temp_c = hourly['temperature_2m'][idx]
                temp_f = round(temp_c * 9/5 + 32)
                humidity = hourly['relative_humidity_2m'][idx]
                rainfall = hourly['precipitation'][idx]
                rainfall_inches = round(rainfall / 25.4, 3)
                return temp_f, humidity, rainfall_inches
            else:
                print("Current time not found in hourly data.")
                return None
        else:
            print(f"Bad response: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error fetching weather: {e}")
        return None

# --- Model Training ---
def train_model(data_path):
    from sklearn.ensemble import HistGradientBoostingRegressor
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()
    season_mapping = {"Winter": 0, "Spring": 1, "Summer": 2, "Fall": 3}
    df['Season_Num'] = df['Season'].map(season_mapping)
    feature_cols = ['Avg_Temp_F', 'Total_Rainfall_Inches', 'Avg_Humidity_Percent', 'Season_Num', 'Population_Density']
    X = df[feature_cols]
    y = df['Usage (GAL)']
    if X['Population_Density'].isnull().any():
        X['Population_Density'].fillna(3200, inplace=True)
    model = HistGradientBoostingRegressor()
    model.fit(X, y)
    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)

# --- Load Model ---
def load_forecast_model():
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
    return model

# --- Find Latest Uploaded File ---
def get_latest_uploaded_file(folder="uploads/"):
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.csv')]
    if not files:
        return None
    latest_file = max(files, key=os.path.getctime)
    return latest_file

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", ["Home", "Upload New Data", "View Forecast"])

# --- Pages ---
if page == "Home":
    st.image("app/static/epwater_logo.png", width=300)
    st.title("El Paso Water Demand Forecast Dashboard")
    st.markdown("""
    Welcome!  
    - 📤 Upload new water usage data  
    - 📈 View predicted water demand by ZIP code based on **live weather**  
    - 🏛️ Get dynamic recommendations for infrastructure planning  
    """, unsafe_allow_html=True)

    with st.expander("About This Project"):
        st.write("""
        This dashboard uses machine learning and real-time weather data to forecast water demand at the zip code level.
        Designed to help optimize pump, tower, and infrastructure planning as El Paso grows.
        """)

elif page == "Upload New Data":
    st.title("📤 Upload New Water Usage Data")
    uploaded_file = st.file_uploader("Upload a water usage CSV file", type=["csv"])
    if uploaded_file is not None:
        st.success(f"Uploaded file: {uploaded_file.name}")
        try:
            save_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"File saved to {save_path}")
            train_model(save_path)
            st.success("✅ Model retrained successfully with new data!")
        except Exception as e:
            st.error(f"Error reading file: {e}")

elif page == "View Forecast":
    st.title("📈 Forecast Water Usage - Live Weather Adjusted")
    latest_uploaded = get_latest_uploaded_file()
    if latest_uploaded:
        data_source = latest_uploaded
        st.success(f"Using latest uploaded file: {os.path.basename(latest_uploaded)}")
    else:
        data_source = CLEAN_BASELINE_FILE
        st.info("No uploaded file found. Using baseline historical dataset.")

    try:
        df = pd.read_csv(data_source)
        df.columns = df.columns.str.strip()
        df['ZIP'] = df['ZIP'].astype(str).str.zfill(5)

        if 'Billing Date' in df.columns:
            df['Billing Date'] = pd.to_datetime(df['Billing Date'], errors='coerce')
            df['MonthYear'] = df['Billing Date'].dt.to_period('M').astype(str)
            billing_date_exists = True
        elif 'Month' in df.columns:
            df['MonthYear'] = df['Month'].astype(str)
            billing_date_exists = True
        else:
            billing_date_exists = False

        zip_codes = sorted(df['ZIP'].dropna().unique())
        selected_zip = st.selectbox("Select ZIP Code to Analyze:", zip_codes)

        live_weather = get_live_weather()

        if live_weather:
            temp_now, humidity_now, rainfall_now = live_weather
            st.subheader(f"🌡️ Current Weather for El Paso")
            st.metric(label="Temperature", value=f"{temp_now}°F")
            st.metric(label="Humidity", value=f"{humidity_now}%")
            st.metric(label="Rainfall", value=f"{rainfall_now} inches")

            season_mapping = {"Winter": 0, "Spring": 1, "Summer": 2, "Fall": 3}
            current_month = datetime.now().month
            if current_month in [12, 1, 2]:
                season = "Winter"
            elif current_month in [3, 4, 5]:
                season = "Spring"
            elif current_month in [6, 7, 8]:
                season = "Summer"
            else:
                season = "Fall"
            season_num = season_mapping.get(season, 2)
            population_density = 3200  # Static fallback

            X_live = pd.DataFrame({
                'Avg_Temp_F': [temp_now],
                'Total_Rainfall_Inches': [rainfall_now],
                'Avg_Humidity_Percent': [humidity_now],
                'Season_Num': [season_num],
                'Population_Density': [population_density]
            })

            model = load_forecast_model()
            predicted_usage = model.predict(X_live)[0]

            st.subheader(f"💧 Predicted Water Usage for ZIP {selected_zip}")
            st.metric(label="Predicted Usage", value=f"{predicted_usage:,.0f} Gallons")

            st.subheader(f"📊 Historical and Forecasted Water Usage for ZIP {selected_zip}")

            if billing_date_exists:
                df_zip = df[df['ZIP'] == selected_zip]
                usage_trend = df_zip.groupby('MonthYear')['Usage (GAL)'].sum().reset_index()
                usage_trend = usage_trend.sort_values('MonthYear')
                last_month = datetime.now().replace(day=1)
                future_months = [(last_month + relativedelta(months=i)).strftime("%Y-%m") for i in range(1, 7)]
                base_prediction = model.predict(X_live)[0]
                future_preds = [base_prediction * np.random.uniform(0.95, 1.05) for _ in range(6)]
                future_df = pd.DataFrame({'MonthYear': future_months, 'Usage (GAL)': future_preds})
                combined_df = pd.concat([usage_trend, future_df]).reset_index(drop=True)
                combined_df = combined_df.sort_values('MonthYear')
                st.line_chart(combined_df.set_index('MonthYear')['Usage (GAL)'])

                st.caption("Historical usage based on billing data. Future usage predicted from live weather conditions.")

            else:
                st.warning("No billing date or month data available for historical trends.")

            st.subheader("🏛️ Infrastructure Advisory Based on Predicted Usage")

            if predicted_usage > 1.5 * df['Usage (GAL)'].mean():
                st.error("⚡ High water demand predicted! Consider scaling tower and pump capacity.")
                st.write("- Activate contingency water sources\n- Expand reserve storage\n- Increase security monitoring")
            elif predicted_usage > df['Usage (GAL)'].mean():
                st.warning("🔔 Moderate increase expected. Monitor closely.")
                st.write("- Pre-stage maintenance\n- Watch energy consumption levels")
            else:
                st.success("✅ Demand within normal range.")
                st.write("- Continue normal operations\n- Monitor ongoing weather impacts.")

        else:
            st.error("🚫 Could not retrieve live weather data. Please refresh the page.")

    except Exception as e:
        st.error(f"Error processing forecast: {e}")

else:
    st.error("Unknown page selection.")
