import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pickle
from llama_cpp import Llama

# --- Setup Upload Directory ---
UPLOAD_FOLDER = "uploads/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Paths ---
CLEAN_BASELINE_FILE = "data/cleaned/water_data_cleaned.csv"
MODEL_FILE = "models/forecast_model.pkl"
LLM_MODEL_FILE = "models/tiny_llama.ggmlv3.q4_0.bin"  # Example small model you can download separately

# --- Load Machine Learning Model ---
@st.cache_resource
def load_forecast_model():
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
    return model

model = load_forecast_model()

# --- Load Local LLM ---
@st.cache_resource
def load_local_llm():
    return Llama(model_path=LLM_MODEL_FILE, n_ctx=512)

llm = load_local_llm()

# --- Pull Live Weather Data ---
def get_live_weather(city="El+Paso"):
    url = f"https://wttr.in/{city}?format=%t|%h|%p"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            temp, humidity, rainfall = response.text.strip().split("|")
            temp = int(temp.replace("+","").replace("°C","").replace("°F",""))
            humidity = int(humidity.replace("%",""))
            rainfall = float(rainfall.replace("mm",""))
            return temp, humidity, rainfall
        else:
            return None
    except Exception:
        return None

# --- Find Latest Uploaded File ---
def get_latest_uploaded_file(folder="uploads/"):
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.csv')]
    if not files:
        return None
    latest_file = max(files, key=os.path.getctime)
    return latest_file

# --- Streamlit App Layout ---
st.set_page_config(page_title="El Paso Water Forecast Dashboard", layout="wide")

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", ["Home", "Upload New Data", "View Forecast", "Infrastructure Alerts"])

# --- Pages ---
if page == "Home":
    st.title("🚰 El Paso Water Demand Forecast Dashboard")
    st.write("""
    Welcome!  
    - Upload new water usage data  
    - View historical usage by ZIP  
    - Forecast future water demand  
    - Get AI-generated infrastructure alerts
    """)

elif page == "Upload New Data":
    st.title("📤 Upload New Water Usage Data")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        st.success(f"Uploaded file: {uploaded_file.name}")
        try:
            df = pd.read_csv(uploaded_file)
            st.subheader("Preview of Uploaded Data")
            st.dataframe(df.head())

            # Save uploaded file
            save_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"File saved to {save_path}")

        except Exception as e:
            st.error(f"Error reading file: {e}")

elif page == "View Forecast":
    st.title("📈 Forecast and Historical Usage")

    latest_uploaded = get_latest_uploaded_file()

    if latest_uploaded:
        st.success(f"Using latest uploaded file: {os.path.basename(latest_uploaded)}")
        data_source = latest_uploaded
    else:
        st.warning("No uploaded file found. Falling back to baseline data.")
        data_source = CLEAN_BASELINE_FILE

    try:
        df = pd.read_csv(data_source)
        df.columns = df.columns.str.strip()
        df['ZIP'] = df['ZIP'].astype(str).str.zfill(5)
        df['Billing Date'] = pd.to_datetime(df['Billing Date'], errors='coerce')
        df = df.dropna(subset=['Usage (GAL)', 'ZIP'])

        today = datetime.today()
        three_months_ago = today - timedelta(days=90)

        df_recent = df[df['Billing Date'] >= three_months_ago]

        if df_recent.empty:
            st.warning("No recent data available.")
        else:
            st.subheader("Water Usage - Past 3 Months")

            zip_codes = sorted(df_recent['ZIP'].unique())
            selected_zip = st.selectbox("Select ZIP Code to View:", zip_codes)

            filtered_df = df_recent[df_recent['ZIP'] == selected_zip]
            filtered_df['Month'] = filtered_df['Billing Date'].dt.to_period('M').astype(str)

            usage_by_month = filtered_df.groupby('Month')['Usage (GAL)'].sum().reset_index()

            fig, ax = plt.subplots(figsize=(10,6))
            ax.plot(usage_by_month['Month'], usage_by_month['Usage (GAL)'], marker='o')
            ax.set_title(f'Water Usage - Past 3 Months (ZIP {selected_zip})')
            ax.set_xlabel('Month')
            ax.set_ylabel('Usage (Gallons)')
            plt.xticks(rotation=45)
            st.pyplot(fig)

        # Forecast next 6 months
        st.subheader("🔮 Forecast - Next 6 Months (City-Wide)")

        live_weather = get_live_weather()
        if live_weather:
            temp_now, humidity_now, rainfall_now = live_weather
            st.info(f"Live Weather: {temp_now}°F, {humidity_now}% Humidity, {rainfall_now}in Rainfall")

            # Create 6-month forecast
            future_months = pd.date_range(start=today + pd.DateOffset(months=1), periods=6, freq='MS').to_period('M').astype(str)
            avg_season = "Summer" if today.month in [6,7,8] else "Winter"  # basic season hack

            season_mapping = {"Winter":0, "Spring":1, "Summer":2, "Fall":3}
            season_num = season_mapping.get(avg_season, 2)

            population_density = 3200  # hardcoded for now

            X_future = pd.DataFrame({
                'Avg_Temp_F': [temp_now]*6,
                'Total_Rainfall_Inches': [rainfall_now]*6,
                'Avg_Humidity_Percent': [humidity_now]*6,
                'Season_Num': [season_num]*6,
                'Population_Density': [population_density]*6
            })

            future_preds = model.predict(X_future)

            forecast_df = pd.DataFrame({
                'Month': future_months,
                'Predicted Usage (GAL)': future_preds
            })

            fig2, ax2 = plt.subplots(figsize=(10,6))
            ax2.plot(forecast_df['Month'], forecast_df['Predicted Usage (GAL)'], marker='x', linestyle='--', color='red', label='Forecast')
            ax2.set_title('Forecasted Water Usage - Next 6 Months')
            ax2.set_xlabel('Month')
            ax2.set_ylabel('Usage (Gallons)')
            ax2.legend()
            plt.xticks(rotation=45)
            st.pyplot(fig2)

        else:
            st.error("Could not retrieve live weather data.")

    except Exception as e:
        st.error(f"Error processing data: {e}")

elif page == "Infrastructure Alerts":
    st.title("🚨 Infrastructure Alerts (AI Powered)")

    try:
        # Use last forecast if available
        if 'forecast_df' in locals():
            prompt = f"""
            As an infrastructure planning assistant, analyze the following water usage forecast:

            {forecast_df.to_string(index=False)}

            Which months should be flagged for high infrastructure strain, and what actionable steps should the city take?
            """

            output = llm(prompt, temperature=0.2, top_p=0.9, max_tokens=400)
            st.subheader("📋 AI Infrastructure Advisory")
            st.write(output['choices'][0]['text'].strip())
        else:
            st.info("Please view the Forecast first to generate predictions.")

    except Exception as e:
        st.error(f"Error generating AI advisory: {e}")

else:
    st.error("Unknown page selection.")
