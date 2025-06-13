from flask import Flask, render_template, request
import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime
import os

app = Flask(__name__)

# --- Paths to your saved model and resources ---
MODEL_PATH = "model/Bengaluru_House_Data.pkl"
LOCATION_JSON = "model/Benguluru_house_data_resources/location.json"
SOCIETY_JSON = "model/Benguluru_house_data_resources/society.json"
AREA_TYPE_JSON = "model/Benguluru_house_data_resources/area_type.json"

# --- Load model and resources ---
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully.")

    with open(LOCATION_JSON, "r") as f:
        locations_full = json.load(f)["locations"] # These are the original full locations
    print("Full Locations loaded successfully.")

    with open(SOCIETY_JSON, "r") as f:
        societies_reduced_unique = json.load(f)["locations"] # These are the reduced societies
    print("Reduced Societies loaded successfully.")

    with open(AREA_TYPE_JSON, "r") as f:
        area_types_unique = json.load(f)["locations"] # These are the area types
    print("Area Types loaded successfully.")

    # Get the exact feature names the model was trained with
    # This is CRUCIAL for building the prediction input correctly
    model_features = model.feature_names_in_.tolist()
    print(f"Model expects {len(model_features)} features.")
    # print("Expected features:", model_features)

except FileNotFoundError as e:
    print(f"Error: Required file not found: {e}")
    print("Please ensure '2nd.py' was run and generated all necessary model and JSON files.")
    exit()
except Exception as e:
    print(f"Error loading model or resources: {e}")
    exit()

# --- Helper functions from 2nd.py (replicated for consistency) ---
def date_category(date_str):
    if (date_str == "Ready To Move" or date_str == "Immediate Possession"):
        return datetime.now().strftime("%d-%m-%Y")
    else:
        date_str += "-2025" # Assuming 2025 for future dates, adjust as needed
        try:
            date_obj = datetime.strptime(date_str, "%d-%b-%Y")
        except ValueError:
            # Fallback for invalid date format, e.g., if user inputs 'invalid-date'
            return datetime.now().strftime("%d-%m-%Y") # Default to today if format is wrong
        return date_obj.strftime("%d-%m-%Y")

# --- Flask Routes ---
@app.route('/')
def index():
    # Pass the unique options to the HTML template for dropdowns
    return render_template("index.html",
                           locations=locations_full, # Use full locations for user selection
                           societies=societies_reduced_unique, # Use reduced societies for dropdown
                           area_types=area_types_unique)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # --- 1. Get raw inputs from the form ---
        sqft = float(request.form["sqft"])
        bath = int(request.form["bath"])
        balcony = int(request.form["balcony"])
        bhk = int(request.form["bhk"])
        area_type = request.form["area_type"]
        location_selected = request.form["location"] # User selects from full locations
        society_selected = request.form["society"] # User selects from reduced societies
        availability_input = request.form["availability"] # New input
        price_per_sqft = float(request.form["price_per_sqft"]) # New input

        # --- 2. Replicate feature engineering from 2nd.py ---

        # 2.1 Calculate availability_days
        availability_date_str = date_category(availability_input)
        availability_date_obj = datetime.strptime(availability_date_str, "%d-%m-%Y")
        days_diff = (availability_date_obj - pd.Timestamp.today()).days
        availability_days = float(max(0, days_diff)) # Ensure non-negative

        # 2.2 Determine location_reduced and society_reduced for the *input*
        # This is where it gets tricky: your model was trained on filtered data.
        # If the user selects a location/society that was filtered out as 'others'/'other'
        # in training, we need to decide how to handle it.
        # For simplicity, we'll assume the loaded JSONs *only* contain what the model saw.
        # If a selected location/society is not in `model_features` (i.e., it was 'others'),
        # the get_dummies will create a column of all zeros for it, which is correct
        # if the model didn't see that specific category.

        # Create a dictionary for the input features
        input_dict = {
            'total_sqft': sqft,
            'bath': float(bath),
            'balcony': float(balcony),
            'bhk': float(bhk),
            'availability_days': availability_days,
            'price_per_sqft': price_per_sqft,
            'area_type': area_type,
            'location_reduced': location_selected, # We're passing the original location for dummy creation
            'society_reduced': society_selected # We're passing the original society for dummy creation
        }

        # Create a DataFrame from the single input
        # Note: We use original column names for dummy creation
        input_df_raw = pd.DataFrame([input_dict])

        # Apply get_dummies using the same columns as in 2nd.py
        # You need to ensure these columns ('area_type', 'location_reduced', 'society_reduced')
        # match the column names in your `input_df_raw` which will be used for get_dummies.
        # For `location_reduced` and `society_reduced`, the input data needs to align with
        # how `df2`'s `location_reduced` and `society_reduced` were defined *before* get_dummies.
        # Given your `2nd.py` saves `unique_locations` (full locations) and `unique_society` (reduced societies),
        # we'll map the user's selected `location_selected` and `society_selected` to the
        # appropriate reduced/filtered category name if they exist in `model_features`.

        # Re-apply the 'others'/'other' reduction logic if needed for the input,
        # otherwise, `get_dummies` will create new columns if the input values
        # for location/society were not in the training set's reduced categories.
        # This is the trickiest part, as your model was trained on a *filtered* dataset.
        # To simplify, we'll rely on `get_dummies` to handle unseen categories by
        # creating all-zero columns for them, assuming the model can handle it.
        # The key is to pass `columns` to get_dummies.

        # For this to work robustly, the `locations_full` in your JSON should be
        # the *original* locations, and the `societies_reduced_unique` should be
        # the *final reduced* societies from `df2` in your `2nd.py`.
        # Your `2nd.py` saves `df["location"].unique().tolist()` as `unique_locations`
        # and `df2["society_reduced"].unique().tolist()` as `unique_society`.
        # This means `locations_full` is *all* locations, and `societies_reduced_unique` are *only* the reduced ones.
        # This makes the dummy creation complicated.

        # Let's adjust the input dict creation to match the columns that `get_dummies` expects:
        # We need to explicitly tell `get_dummies` which columns to create for the categorical features
        # based on *all* possible categories (from `model_features`).

        # Create a dummy DataFrame with all possible dummy columns initialized to zero
        # This ensures the input DataFrame for prediction has the exact same columns as model.feature_names_in_
        dummy_df_structure = pd.DataFrame(columns=model_features)
        
        # Populate the numerical columns
        for col in ['total_sqft', 'bath', 'balcony', 'bhk', 'availability_days', 'price_per_sqft']:
            if col in dummy_df_structure.columns:
                dummy_df_structure.loc[0, col] = input_dict[col]

        # Populate the categorical columns
        # For area_type
        area_type_col = f'area_type_{area_type}'
        if area_type_col in dummy_df_structure.columns:
            dummy_df_structure.loc[0, area_type_col] = 1.0

        # For location_reduced (this is the problematic one due to 'others' removal)
        # We need to find the `location_reduced` form of `location_selected`.
        # If `location_selected` is one of the `location_reduced` categories the model saw,
        # then set that dummy. Otherwise, it should effectively be an 'others' (all zeros for location dummies).
        # Since your `2nd.py` *dropped* 'others', the model never saw a specific column for 'others'.
        # The safest is to ensure `locations_full` in `location.json` matches the actual
        # `location_reduced` categories used during training.
        # Assuming your `location.json` now correctly contains `location_reduced` values (not original ones).
        location_reduced_col = f'location_reduced_{location_selected}'
        if location_reduced_col in dummy_df_structure.columns:
            dummy_df_structure.loc[0, location_reduced_col] = 1.0

        # For society_reduced
        society_reduced_col = f'society_reduced_{society_selected}'
        if society_reduced_col in dummy_df_structure.columns:
            dummy_df_structure.loc[0, society_reduced_col] = 1.0

        # Fill NaN with 0.0 for any unassigned dummy columns (which will be most of them)
        input_for_prediction = dummy_df_structure.fillna(0.0).iloc[[0]] # Get the first row as DataFrame

        # Ensure the column order is exactly as expected by the model
        input_for_prediction = input_for_prediction[model_features]

        # Debugging: Print the final input DataFrame and its shape
        print("\nInput DataFrame for prediction:")
        print(input_for_prediction)
        print("Shape of input DataFrame:", input_for_prediction.shape)
        print("Expected model features length:", len(model_features))


        # --- 3. Predict ---
        prediction = model.predict(input_for_prediction)[0]

        # --- 4. Display Result ---
        return render_template("index.html",
                               locations=locations_full,
                               societies=societies_reduced_unique,
                               area_types=area_types_unique,
                               prediction_text=f"Predicted Price: ₹ {prediction:.2f} Lakhs")

    except ValueError:
        return render_template("index.html",
                               locations=locations_full,
                               societies=societies_reduced_unique,
                               area_types=area_types_unique,
                               prediction_text="Error: Please enter valid numerical values.")
    except Exception as e:
        # Catch any other unexpected errors during prediction
        return render_template("index.html",
                               locations=locations_full,
                               societies=societies_reduced_unique,
                               area_types=area_types_unique,
                               prediction_text=f"Prediction Error: {str(e)}")

if __name__ == '__main__':
    # Ensure the 'model' directory exists
    os.makedirs('model/Benguluru_house_data_resources', exist_ok=True)
    app.run(debug=True)