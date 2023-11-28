import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import math
import tkinter as tk
from tkinter import simpledialog
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import joblib

new_df = pd.read_csv("C:\\Users\\kille\\Downloads\\Debajit_Raw_training_Data_Training_9_51.csv")

# Load the aggregated model 'rf_model_0_100'
rf_model_0_100 = joblib.load('C:\\Users\\kille\\Downloads\\rf_model_0_100_tuned.pkl')

# Define the selected features for the model
selected_features = ['odometer', 'temperature', 'avg_bat_voltage', 'charge_start_soc', 'charge_end_soc']

numeric_columns = ['Bat_A_iCurrent', 'Bat_B_iCurrent', 'Bat_A_iVoltage', 'Bat_B_iVoltage']
new_df[numeric_columns] = new_df[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Now perform the calculations
new_df['avg_bat_current'] = (new_df['Bat_A_iCurrent'] + new_df['Bat_B_iCurrent']) / 2
new_df['avg_bat_voltage'] = (new_df['Bat_A_iVoltage'] + new_df['Bat_B_iVoltage']) / 2

# Remove rows where avg_bat_current <= -3 or avg_bat_current >= 3
new_df_filtered = new_df[(new_df['avg_bat_current'] > -3) & (new_df['avg_bat_current'] < 3)].copy()

new_df_filtered.rename(columns={'soc': 'charge_start_soc'}, inplace=True)
new_df_filtered['charge_end_soc'] = 100

new_df_filtered['average_soc_duration'] = new_df_filtered['time_estimation_instant'] / (new_df_filtered.apply(lambda row: math.ceil((row['charge_end_soc'] - row['charge_start_soc']) / 5), axis=1))

# Create a copy of the original DataFrame
new_df_no_outliers = new_df_filtered.copy()

# Fill missing values with 0 in the entire dataframe
new_df_filled = new_df_no_outliers.fillna(0)

# Remove rows where 'average_soc_duration' is infinity
new_df_filled = new_df_filled.replace([np.inf, -np.inf], np.nan)
new_df_filled = new_df_filled.dropna(subset=['average_soc_duration'])

# Define the maximum number of records with charge_end_soc == 100
max_records_with_100 = 5

# Create a list of ranges for 'time_estimation_instant'
ranges = [(0, 5000), (5001, 10000), (10001, 15000), (15001, 20000)]
dataframes = []
sample_df = pd.DataFrame()

# Randomly select records in each range
for (start, end) in ranges:
    df_range = new_df_filled[(new_df_filled['time_estimation_instant'] >= start) & (new_df_filled['time_estimation_instant'] <= end)]
    dataframes.append(df_range.sample(n=10, random_state=42))

sample_df = pd.concat(dataframes)

# Reset the index and start from 1
sample_df.reset_index(drop=True, inplace=True)
sample_df.index += 1  # This will start the index from 1

# Set the page layout to split into two halves vertically
st.set_page_config(layout="wide")

# Create a Streamlit sidebar on the left half
st.sidebar.header("Sidebar")

# Display the DataFrame on the left half
st.sidebar.write("Sample DataFrame:")
st.sidebar.write(sample_df)

# Display additional information in a tabular format
evaluation_metrics = {
    "Metric": ["MAE", "MAPE"],
    "Value": ["628.1320 seconds", "15.2578 %"]
}

st.sidebar.write("Evaluation Metrics:")
st.sidebar.table(evaluation_metrics)

# Create checkboxes for each row
selected_rows = st.sidebar.multiselect("Select rows:", sample_df.index.tolist())

# Display the selected rows as a DataFrame on the right half
if selected_rows:
    selected_data = sample_df.loc[selected_rows]
    st.subheader("Selected Record:")
    st.write(selected_data)

    # Display required information below the selected record
    st.subheader("Values corresponding to the features that are required to construct the model :")
    additional_info = {
        'Charging Start Odometer': [selected_data['odometer'].values[0]],
        'Average Battery Temperature': [selected_data['temperature'].values[0]],
        'Average Battery Voltage': [selected_data['avg_bat_voltage'].values[0]],
        'Charging Start Percentage': [selected_data['charge_start_soc'].values[0]],
        'Charging End Percentage': [selected_data['charge_end_soc'].values[0]],
    }
    additional_info_df = pd.DataFrame(additional_info)
    st.table(additional_info_df)

    # Prepare the feature matrix X for the testing data and target vector y
    X_test = new_df_filled[selected_features]
    y_test = new_df_filled['time_estimation_instant']

    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'n_estimators': [25, 30, 35],
        'max_depth': [7, 9, 11],
        'min_samples_split': [15, 20, 25]
    }

    # Create a Random Forest Regression model
    rf_model_0_100 = RandomForestRegressor(random_state=42)

    # Perform GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(rf_model_0_100, param_grid, scoring='neg_mean_absolute_error', cv=5)
    grid_search.fit(X_test, y_test)

    # Get the best hyperparameters
    best_params = grid_search.best_params_

    # Get the best model from GridSearchCV
    best_rf_model = grid_search.best_estimator_

    def calculate_predicted_time(selected_data):
        # Ensure that 'selected_data' contains the selected features in the right order
        selected_data = selected_data[selected_features].values.reshape(1, -1)

        # Use the best model to make predictions for this single record
        y_pred_sum = best_rf_model.predict(selected_data)

        return y_pred_sum[0]

    def calculate_predicted_time_manual(selected_data, charge_end_soc_manual):
        # Get the 'charge_end_soc' value
        end_per = charge_end_soc_manual if charge_end_soc_manual is not None else \
        selected_data['charge_end_soc'].values[0]

        # Create a new record with the updated 'charge_end_soc'
        record = selected_data.copy()
        record['charge_end_soc'] = end_per
        record = record[selected_features].values.reshape(1, -1)

        # Use the best model to make predictions for this single record
        y_pred_sum = best_rf_model.predict(record)

        return y_pred_sum[0]

    # Display the "PREDICT" button
    if st.button("PREDICT"):

        # Calculate the predicted charging time
        predicted_time = calculate_predicted_time(selected_data)

        # Check if predicted_time is a Series
        if isinstance(predicted_time, pd.Series):
            # Extract the first value from the Series
            predicted_time = predicted_time.iloc[0]

        if predicted_time is not None:
            # Format and display the predicted time
            st.markdown(f"**Total Predicted Charging time:** {predicted_time:.2f} seconds")
        else:
            st.warning("Prediction failed. Please check the selected data and model.")

    # Initialize charge_end_soc_manual as a global variable using st.session_state
    if 'charge_end_soc_manual' not in st.session_state:
        st.session_state.charge_end_soc_manual = None

    if st.button("MANUAL charge_end_soc INPUT"):
        # Use a simple Python input dialog to get the charge_end_soc_manual value
        root = tk.Tk()
        root.withdraw()
        charge_end_soc_manual_input = simpledialog.askinteger(
            "Manual charge_end_soc INPUT",
            "Enter charge_end_soc (integer) : ",
            minvalue=selected_data['charge_start_soc'].min(),
            maxvalue=100
        )

        if charge_end_soc_manual_input is not None:
            st.session_state.charge_end_soc_manual = charge_end_soc_manual_input
            print(f"Manual charge_end_soc_input: {charge_end_soc_manual_input}")

    if st.button("Calculate Predicted Charging Time"):
        if st.session_state.charge_end_soc_manual is not None:
            st.write(f"charge_end_soc_manual: {st.session_state.charge_end_soc_manual}")  # Display the manual input
            record = selected_data.copy()
            record['charge_end_soc'] = st.session_state.charge_end_soc_manual
            record = record[selected_features].values.reshape(1, -1)

            # Use the model to make predictions
            predicted_time_manual = best_rf_model.predict(record)[0]
            st.write(f"predicted_time_manual: {predicted_time_manual:.2f} seconds")  # Display the predicted time

            # Display the predicted time
            st.markdown(f"**Total Predicted Charging time:** {predicted_time_manual:.2f} seconds")
        else:
            st.warning("Please enter a valid integer for charge_end_soc before calculating.")
