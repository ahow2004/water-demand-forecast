import streamlit as st
import pandas as pd
import os

# --- Create Upload Directory if Needed ---
UPLOAD_FOLDER = "uploads/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", ["Home", "Upload New Data", "View Forecast", "Infrastructure Alerts"])

# --- Pages ---
if page == "Home":
    st.title("🚰 El Paso Water Demand Forecast Dashboard")
    st.write("""
    Welcome to the dashboard! 
    - Upload new water usage data
    - View past 3 months usage
    - Forecast next 6 months usage
    - Get actionable infrastructure insights
    """)
    
elif page == "Upload New Data":
    st.title("📤 Upload New Water Usage Data")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        # Show file name
        st.success(f"Uploaded file: {uploaded_file.name}")
        
        # Read CSV into DataFrame
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
    st.write("Graphs and predictions will appear here.")

elif page == "Infrastructure Alerts":
    st.title("🚨 Infrastructure Alerts")
    st.write("Spike risk analysis and recommendations here.")
