import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# --- Page Configuration ---
st.set_page_config(
    page_title="NYC Taxi Fare Predictor",
    page_icon="🚖",
    layout="wide"
)

# --- Title and Introduction ---
st.title("🚖 NYC Taxi Fare Prediction: 4-Model Showdown")
st.markdown("""
**Objective:** Predict taxi fares using 4 different Machine Learning algorithms and determine the most accurate one.\n
**Models Compared:**
1.  **Linear Regression** (Baseline)
2.  **Decision Tree** (Simple non-linear)
3.  **Random Forest** (Ensemble bagging)
4.  **HistGradientBoosting** (Ensemble boosting)
""")

# --- 1. Data Loading & Cleaning Function ---
@st.cache_data
def load_and_clean_data(filepath):
    try:
        # Load data (using low_memory=False to handle mixed types warning)
        df = pd.read_csv(filepath, low_memory=False)
    except FileNotFoundError:
        return None, 0, 0

    original_count = len(df)

    # 1. Robust Date Parsing
    df['lpep_pickup_datetime'] = pd.to_datetime(df['lpep_pickup_datetime'], errors='coerce')
    df['lpep_dropoff_datetime'] = pd.to_datetime(df['lpep_dropoff_datetime'], errors='coerce')

    # 2. Robust Numeric Conversion
    cols_to_numeric = ['trip_distance', 'total_amount', 'PULocationID', 'DOLocationID', 'passenger_count']
    for col in cols_to_numeric:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 3. Drop Nulls in critical columns
    df = df.dropna(subset=['lpep_pickup_datetime', 'lpep_dropoff_datetime', 'trip_distance', 'total_amount'])

    # 4. Fill Missing
    df['passenger_count'] = df['passenger_count'].fillna(1)

    # 5. Filter Outliers
    # Fares between $2.50 and $500
    df = df[(df['total_amount'] >= 2.50) & (df['total_amount'] < 500)]
    # Distances between 0.1 and 200 miles
    df = df[(df['trip_distance'] > 0.1) & (df['trip_distance'] < 200)]

    # 6. Feature Engineering
    # Duration in minutes
    df['trip_duration_min'] = (df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime']).dt.total_seconds() / 60
    # Valid duration check (1 min to 5 hours)
    df = df[(df['trip_duration_min'] >= 1) & (df['trip_duration_min'] <= 300)]
    
    # Time components
    df['pickup_hour'] = df['lpep_pickup_datetime'].dt.hour
    df['pickup_day_of_week'] = df['lpep_pickup_datetime'].dt.dayofweek # 0=Mon, 6=Sun

    cleaned_count = len(df)
    return df, original_count, cleaned_count

# --- Main App Logic ---

# Load Data
with st.spinner('Loading and Cleaning Dataset...'):
    df_clean, rows_raw, rows_clean = load_and_clean_data('2023_Green_Taxi_Trip_Data.csv')

if df_clean is not None:
    # --- Sidebar Inputs ---
    st.sidebar.header("📝 Trip Details")
    
    input_dist = st.sidebar.number_input("Distance (miles)", 0.1, 100.0, 3.5, 0.1)
    input_time = st.sidebar.number_input("Duration (minutes)", 1.0, 300.0, 15.0, 1.0)
    input_pass = st.sidebar.selectbox("Passengers", [1, 2, 3, 4, 5, 6])
    input_hour = st.sidebar.slider("Pickup Hour (0-23)", 0, 23, 14)
    input_day = st.sidebar.selectbox("Day of Week", options=[0,1,2,3,4,5,6], 
                                     format_func=lambda x: ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][x])
    input_pu = st.sidebar.number_input("Pickup Location ID", 1, 265, 42)
    input_do = st.sidebar.number_input("Dropoff Location ID", 1, 265, 74)

    st.sidebar.markdown("---")
    st.sidebar.info(f"Using {rows_clean:,} cleaned records for training.")

    # --- Model Training Section ---
    st.header("1. Model Evaluation & Comparison")
    
    # 1. Prepare Data
    features = ['trip_distance', 'trip_duration_min', 'pickup_hour', 'pickup_day_of_week', 'passenger_count', 'PULocationID', 'DOLocationID']
    target = 'total_amount'
    
    X = df_clean[features]
    y = df_clean[target]
    
    # Performance Optimization: Sample data if too large for live training
    # Training Random Forest on 700k rows in a web app might timeout, so we sample if necessary
    if len(X) > 100000:
        X_sample = X.sample(100000, random_state=42)
        y_sample = y.loc[X_sample.index]
        X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Define Models
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=20, max_depth=10, random_state=42, n_jobs=-1),
        "Gradient Boosting": HistGradientBoostingRegressor(max_iter=50, random_state=42)
    }

    # 3. Train & Evaluate
    results = []
    trained_models = {}

    progress_bar = st.progress(0)
    
    for i, (name, model) in enumerate(models.items()):
        # Train
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        results.append({
            "Model": name,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2 Score": r2
        })
        progress_bar.progress((i + 1) / len(models))

    # 4. Display Results DataFrame
    results_df = pd.DataFrame(results).set_index("Model")
    
    # Find Best Model (Min RMSE)
    best_model_name = results_df['RMSE'].idxmin()
    best_model_r2 = results_df.loc[best_model_name, 'R2 Score']
    
    # Style the dataframe
    st.dataframe(results_df.style.highlight_min(subset=['MAE', 'MSE', 'RMSE'], color='#d4edda')
                             .highlight_max(subset=['R2 Score'], color='#d4edda')
                             .format("{:.4f}"))

    st.success(f"🏆 **Winner:** The **{best_model_name}** is the most accurate model with the lowest RMSE ({results_df.loc[best_model_name, 'RMSE']:.2f}) and highest R² ({best_model_r2:.2f}).")

    st.divider()

    # --- Prediction Section ---
    st.header("2. Live Prediction Dashboard")
    
    # User Input Dataframe
    user_input = pd.DataFrame({
        'trip_distance': [input_dist],
        'trip_duration_min': [input_time],
        'pickup_hour': [input_hour],
        'pickup_day_of_week': [input_day],
        'passenger_count': [input_pass],
        'PULocationID': [input_pu],
        'DOLocationID': [input_do]
    })
    
    # Generate predictions from all models
    preds = {}
    for name, model in trained_models.items():
        preds[name] = model.predict(user_input)[0]

    # Display Cards
    cols = st.columns(len(models))
    for idx, (name, pred) in enumerate(preds.items()):
        with cols[idx]:
            if name == best_model_name:
                st.markdown(f"### ⭐ {name}")
                st.markdown(f"<h1 style='color: #28a745;'>${pred:.2f}</h1>", unsafe_allow_html=True)
                st.caption("Recommended (Most Accurate)")
            else:
                st.markdown(f"### {name}")
                st.markdown(f"<h2 style='color: #6c757d;'>${pred:.2f}</h2>", unsafe_allow_html=True)

    # --- Visualization ---
    st.divider()
    st.header("3. Error Analysis")
    
    tab1, tab2 = st.tabs(["R² Score Comparison", "Accuracy vs Model Complexity"])
    
    with tab1:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=results_df.index, y=results_df['R2 Score'], palette="viridis", ax=ax)
        plt.ylim(0, 1.0)
        plt.title("Model Accuracy (R² Score)")
        plt.ylabel("R² Score (Higher is better)")
        st.pyplot(fig)
        
    with tab2:
        st.info("The chart above shows how much variance in the taxi fare each model can explain. Gradient Boosting and Random Forest typically outperform simple Linear Regression because they handle complex traffic patterns better.")

else:
    st.error("Please upload the '2023_Green_Taxi_Trip_Data.csv' file.")



















