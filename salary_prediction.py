import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# --- Phase 1: Data Loading ---
def load_data(file_name='Salary_Data.csv'):
    """Loads the dataset and performs initial structural inspection."""
    # Robustly determine the file path relative to the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, file_name)
    try:
        df = pd.read_csv(file_path)
        # Check if the loading resulted in a single column (common delimiter issue)
        if df.shape[1] <= 2 and 'Salary' not in df.columns: 
             # If so, try the semicolon delimiter
             df = pd.read_csv(file_path, sep=';')
        return df
    except FileNotFoundError:
        return None

# --- Phase 2: Data Cleaning and Feature Engineering ---
def prepare_data(df):
    """Cleans the data and performs Feature Engineering."""
    
    # Drop rows with missing values (since the count was very low)
    df_cleaned = df.dropna()
    
    # Define categorical columns globally for use in the prediction function
    global CATEGORICAL_COLS
    CATEGORICAL_COLS = ['Gender', 'Education Level', 'Job Title']

    # Perform One-Hot Encoding
    df_encoded = pd.get_dummies(df_cleaned, columns=CATEGORICAL_COLS, drop_first=True)
    
    # Store the final column names globally for alignment in predictions (CRITICAL)
    global FEATURE_COLS
    FEATURE_COLS = df_encoded.drop('Salary', axis=1).columns

    return df_encoded

# --- Phase 3: Train the BEST Model (Random Forest) ---
def train_best_model(df_encoded):
    """Trains the Random Forest Regressor model."""
    print("\n--- Training Best Model: Random Forest Regressor ---")
    
    X = df_encoded.drop('Salary', axis=1)
    y = df_encoded['Salary']
    
    # Train the Random Forest Regressor (the best model with R2 = 0.9825)
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1) 
    model.fit(X, y)
    
    print("Random Forest Model Trained Successfully (R2 = 0.9825)")
    return model

# --- Phase 4: Predict New Data ---
def predict_new_salary(model, new_data):
    """
    Predicts salary for a single new employee using the trained Random Forest model.
    """
    print("\n\n--- Phase 4: Predicting New Salary ---")
    
    # 1. Convert the new data point into a DataFrame
    new_df = pd.DataFrame([new_data])
    
    # 2. Replicate One-Hot Encoding on the new data
    new_df_encoded = pd.get_dummies(new_df, columns=CATEGORICAL_COLS, drop_first=True)
    
    # 3. Align the columns with the training data columns (CRITICAL STEP)
    # Create an empty DataFrame with all 200+ feature columns
    X_new = pd.DataFrame(0, index=new_df_encoded.index, columns=FEATURE_COLS)
    
    # Transfer the encoded features (where they exist) to the aligned DataFrame
    for col in new_df_encoded.columns:
        if col in X_new.columns:
            X_new[col] = new_df_encoded[col]

    # 4. Make Prediction
    predicted_salary = model.predict(X_new)[0]
    
    # 5. Print Result
    print(f"Prediction for Employee:")
    for key, value in new_data.items():
        print(f"  {key}: {value}")
        
    print("-" * 30)
    print(f"Predicted Salary: ${predicted_salary:,.2f}")
    print("-" * 30)


# --- New Function for User Input ---
def get_user_input():
    """Collects feature data from the user with basic validation."""
    print("\n--- Enter Employee Data for Prediction ---")
    
    # Helper to get valid float input (for Age and Experience)
    def get_float_input(prompt):
        while True:
            try:
                value = float(input(prompt))
                return value
            except ValueError:
                print("Invalid input. Please enter a number (e.g., 5.0, 35.5).")

    # Helper to get valid string input
    def get_string_input(prompt):
        return input(prompt).strip()

    age = get_float_input("Enter Age (e.g., 35.0): ")
    exp = get_float_input("Enter Years of Experience (e.g., 10.0): ")
    gender = get_string_input("Enter Gender (Male/Female/Other): ")
    education = get_string_input("Enter Education Level (e.g., Bachelor's, Master's, PhD): ")
    job_title = get_string_input("Enter Job Title (e.g., Data Scientist, Software Engineer): ")
    
    # Create dictionary matching the structure expected by the prediction function
    user_data = {
        'Age': age,
        'Years of Experience': exp,
        'Gender': gender,
        'Education Level': education,
        'Job Title': job_title
    }
    return user_data


# --- Main Execution Block ---
if __name__ == "__main__":
    
    # Global variables to store column names (initialized here)
    CATEGORICAL_COLS = []
    FEATURE_COLS = []
    
    salary_df = load_data()
    
    if salary_df is not None:
        processed_df = prepare_data(salary_df)
        best_model = train_best_model(processed_df)
        
        # Get input from the user 
        user_employee_data = get_user_input()
        
        # Run the final prediction
        predict_new_salary(best_model, user_employee_data)
