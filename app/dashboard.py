import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import numpy as np

# --- Setup Upload Directory ---
UPLOAD_FOLDER = "uploads/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Paths ---
CLEAN_BASELINE_FILE = "data/cleaned/water_data_cleaned.csv"

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
    Welcome to the El Paso Water Forecast Dashboard!

    - 📤 Upload new water usage CSV files
    - 📈 View historical usage trends over the past 3 months
    - 🔮 Forecast future demand and detect usage spikes
    - 🛠️ Get actionable recommendations for infrastructure management
    """)

elif page == "Upload New Data":
    st.title("📤 Upload New Water Usage Data")

    uploaded_file = st.file_uploader("Upload a water usage CSV file", type=["csv"])

    if uploaded_file is not None:
        st.success(f"Uploaded file: {uploaded_file.name}")

        try:
            df = pd.read_csv(uploaded_file)
            st.subheader("Preview of Uploaded Data")
            st.dataframe(df.head())

            # Save uploaded file to uploads/ folder
            save_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.success(f"File saved to {save_path}")

        except Exception as e:
            st.error(f"Error reading file: {e}")

elif page == "View Forecast":
    st.title("📈 Forecast and Historical Usage")

    # --- Determine Data Source ---
    latest_uploaded = get_latest_uploaded_file()

    if latest_uploaded:
        st.success(f"Using latest uploaded file: {os.path.basename(latest_uploaded)}")
        data_source = latest_uploaded
    else:
        st.warning("No uploaded file found. Falling back to baseline data.")
        data_source = CLEAN_BASELINE_FILE

    # --- Load Data ---
    try:
        df = pd.read_csv(data_source)

        # --- Minimal Cleaning ---
        df.columns = df.columns.str.strip()
        df['ZIP'] = df['ZIP'].astype(str).str.zfill(5)

        if 'Billing Date' in df.columns:
            df['Billing Date'] = pd.to_datetime(df['Billing Date'], errors='coerce')

        df = df.dropna(subset=['Usage (GAL)', 'ZIP'])

        # --- Filter Last 3 Months ---
        today = datetime.today()
        three_months_ago = today - timedelta(days=90)

        if 'Billing Date' in df.columns:
            df_recent = df[df['Billing Date'] >= three_months_ago]
        else:
            df_recent = df.copy()

        if df_recent.empty:
            st.warning("No recent data available.")
        else:
            st.subheader("Water Usage - Past 3 Months")

            if 'Billing Date' in df_recent.columns:
                df_recent['Month'] = df_recent['Billing Date'].dt.to_period('M').astype(str)
            else:
                df_recent['Month'] = "Unknown"

            usage_by_zip_month = df_recent.groupby(['ZIP', 'Month'])['Usage (GAL)'].sum().reset_index()

            # --- Plot Historical Usage ---
            fig, ax = plt.subplots(figsize=(10,6))

            for zip_code in usage_by_zip_month['ZIP'].unique():
                zip_data = usage_by_zip_month[usage_by_zip_month['ZIP'] == zip_code]
                ax.plot(zip_data['Month'], zip_data['Usage (GAL)'], marker='o', label=f'ZIP {zip_code}')

            ax.set_title('Water Usage by ZIP - Past 3 Months')
            ax.set_xlabel('Month')
            ax.set_ylabel('Usage (Gallons)')
            ax.legend()
            plt.xticks(rotation=45)

            st.pyplot(fig)

        # --- Simple Forecast for Next 6 Months ---
        st.subheader("🔮 Forecast - Next 6 Months Usage")

        if 'Billing Date' in df.columns:
            df['Month'] = df['Billing Date'].dt.to_period('M').astype(str)
        else:
            df['Month'] = "Unknown"

        usage_by_month = df.groupby('Month')['Usage (GAL)'].sum().reset_index()
        usage_by_month['Month_Num'] = np.arange(len(usage_by_month))

        X = usage_by_month[['Month_Num']]
        y = usage_by_month['Usage (GAL)']

        model = LinearRegression()
        model.fit(X, y)

        future_month_nums = np.arange(len(usage_by_month), len(usage_by_month) + 6).reshape(-1,1)
        future_usage_preds = model.predict(future_month_nums)

        future_months = pd.date_range(start=today + pd.DateOffset(months=1), periods=6, freq='MS').to_period('M').astype(str)

        forecast_df = pd.DataFrame({
            'Month': future_months,
            'Predicted Usage (GAL)': future_usage_preds
        })

        fig2, ax2 = plt.subplots(figsize=(10,6))

        ax2.plot(usage_by_month['Month'], usage_by_month['Usage (GAL)'], marker='o', label="Historical Usage")
        ax2.plot(forecast_df['Month'], forecast_df['Predicted Usage (GAL)'], marker='x', linestyle='--', label="Forecasted Usage")

        ax2.set_title('Forecasted Water Usage - Next 6 Months')
        ax2.set_xlabel('Month')
        ax2.set_ylabel('Usage (Gallons)')
        ax2.legend()
        plt.xticks(rotation=45)

        st.pyplot(fig2)

    except Exception as e:
        st.error(f"Error processing data: {e}")

elif page == "Infrastructure Alerts":
    st.title("🚨 Infrastructure Alerts")

    st.write("""
    Risk analysis and pump/tower optimization suggestions will appear here
    based on upcoming forecasted high-usage periods.
    (Coming soon!)
    """)

else:
    st.error("Unknown page selection.")
