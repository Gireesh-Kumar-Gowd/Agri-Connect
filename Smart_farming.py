# -*- coding: utf-8 -*-
"""
Created on Sun Aug 17 15:39:38 2025

@author: Gireesh
"""

# ==============================
# Importing Libraries
# ==============================
import pickle
import joblib
import streamlit as st
from streamlit_option_menu import option_menu


# ==============================
# Load Models
# ==============================

# Crop Recommendation Model (Pickle - binary)
crop_recommendation_model = pickle.load(
    open(r"C:\Users\HP\OneDrive\Documents\projects\AgriConnect\Crop_recommendation.sav", "rb")
)

# --- Load the compressed Random Forest model ---
yield_prediction_model = joblib.load(
    r"C:\Users\HP\OneDrive\Documents\projects\AgriConnect\Yield_predictor_compressed.joblib"
)


# Price Prediction Model (Pickle - binary)
#price_prediction_model = pickle.load(
 #   open(r"C:\Users\HP\OneDrive\Documents\projects\AgriConnect\Price_predictor.sav", "rb")
#)

# ==============================
# Load Scalers & Encoders
# ==============================

# Crop Recommendation
scaler_cr = pickle.load(
    open(r"C:\Users\HP\OneDrive\Documents\projects\AgriConnect\Scaler(CR).sav", "rb")
)
label_encoder_cr = joblib.load(
    r"C:\Users\HP\OneDrive\Documents\projects\AgriConnect\label_encoder(CR).joblib"
)

# Yield Prediction

scaler_yp = joblib.load(
    r"C:\Users\HP\OneDrive\Documents\projects\AgriConnect\Scaler(YP).joblib"
)

# Price Prediction
scaler_pp = joblib.load(
    r"C:\Users\HP\OneDrive\Documents\projects\AgriConnect\Scaler(YP).joblib"
)
label_encoder_pp = joblib.load(
    r"C:\Users\HP\OneDrive\Documents\projects\AgriConnect\label_encoder(YP).joblib"
)


# Sidebar menu
with st.sidebar:
    selected = option_menu(
        'AgriConnect',
        ['Crop Recommendation', 'Yield Prediction', 'Price Prediction'],
        icons=['flower1', 'bar-chart-line', 'currency-rupee'],
        default_index=0
    )

# =============================911
# Crop Recommendation Page
# =============================

if selected == "Crop Recommendation":
    
    st.title("🌱 Crop Recommendation System")
    st.write("Enter soil and climate details to get the best crop suggestion.")

    # Creating input fields
    col1, col2, col3 = st.columns(3)

    with col1:
        N = st.text_input("Nitrogen (N)", "0")

    with col2:
        P = st.text_input("Phosphorus (P)", "0")

    with col3:
        K = st.text_input("Potassium (K)", "0")

    with col1:
        temperature = st.text_input("Temperature (°C)", "0")

    with col2:
        humidity = st.text_input("Humidity (%)", "0")

    with col3:
        ph = st.text_input("pH Level", "0")

    rainfall = st.text_input("Rainfall (mm)", "0")

    crop_result = ""
    
    
   # --- Prediction ---
    crop_result = ""
    if st.button("🌾 Recommend Crop"):
        try:
            # Convert input to float
            user_input = [[float(N), float(P), float(K),
                           float(temperature), float(humidity),
                           float(ph), float(rainfall)]]

            # Scale input
            user_input_scaled = scaler_cr.transform(user_input)

            # Predict (encoded label)
            prediction_encoded = crop_recommendation_model.predict(user_input_scaled)

            # Decode to crop name
            prediction_label = label_encoder_cr.inverse_transform(prediction_encoded)

            crop_result = f"✅ Recommended Crop: **{prediction_label[0]}**"

        except Exception as e:
            crop_result = f"⚠️ Error: {str(e)}"

    st.success(crop_result)
    
# =============================
# Yield Prediction Page
# =============================
elif selected == "Yield Prediction":
    st.title("🌾 Yield Prediction using ML")
    st.write("Enter crop, season, and environmental details to estimate yield per hectare.")

    # --- Fixed mappings (as per your provided encoding) ---
    crop_mapping = {
    'Arecanut': 0, 'Arhar/Tur': 1, 'Bajra': 2, 'Banana': 3, 'Barley': 4,
    'Black pepper': 5, 'Cardamom': 6, 'Cashewnut': 7, 'Castor seed': 8,
    'Coconut': 9, 'Coriander': 10, 'Cotton(lint)': 11, 'Cowpea(Lobia)': 12,
    'Dry chillies': 13, 'Garlic': 14, 'Ginger': 15, 'Gram': 16, 'Groundnut': 17,
    'Guar seed': 18, 'Horse-gram': 19, 'Jowar': 20, 'Jute': 21, 'Khesari': 22,
    'Linseed': 23, 'Maize': 24, 'Masoor': 25, 'Mesta': 26,
    'Moong(Green Gram)': 27, 'Moth': 28, 'Niger seed': 29, 'Oilseeds total': 30,
    'Onion': 31, 'other oilseeds': 32, 'Other Cereals': 33,
    'Other Kharif pulses': 34, 'Other Rabi pulses': 35,
    'Other Summer Pulses': 36, 'Peas & beans (Pulses)': 37, 'Potato': 38,
    'Ragi': 39, 'Rapeseed &Mustard': 40, 'Rice': 41, 'Safflower': 42,
    'Sannhamp': 43, 'Sesamum': 44, 'Small millets': 45, 'Soyabean': 46,
    'Sugarcane': 47, 'Sunflower': 48, 'Sweet potato': 49, 'Tapioca': 50,
    'Tobacco': 51, 'Turmeric': 52, 'Urad': 53, 'Wheat': 54
    }

    season_mapping = {
    'Autumn': 0, 'Kharif': 1, 'Rabi': 2,
    'Summer': 3, 'Whole Year': 4, 'Winter': 5
    }

    state_mapping = {
    'Andhra Pradesh': 0, 'Arunachal Pradesh': 1, 'Assam': 2, 'Bihar': 3,
    'Chhattisgarh': 4, 'Delhi': 5, 'Goa': 6, 'Gujarat': 7, 'Haryana': 8,
    'Himachal Pradesh': 9, 'Jammu and Kashmir': 10, 'Jharkhand': 11,
    'Karnataka': 12, 'Kerala': 13, 'Madhya Pradesh': 14, 'Maharashtra': 15,
    'Manipur': 16, 'Meghalaya': 17, 'Mizoram': 18, 'Nagaland': 19, 'Odisha': 20,
    'Puducherry': 21, 'Punjab': 22, 'Sikkim': 23, 'Tamil Nadu': 24,
    'Telangana': 25, 'Tripura': 26, 'Uttar Pradesh': 27, 'Uttarakhand': 28,
    'West Bengal': 29
    }

    # --- Lists for dropdowns ---
    crop_list   = list(crop_mapping.keys())
    season_list = list(season_mapping.keys())
    state_list  = list(state_mapping.keys())

    # --- Input fields ---
    col1, col2, col3 = st.columns(3)

    with col1:
        crop = st.selectbox("Crop", ["Select Crop"] + crop_list)

    with col2:
        season = st.selectbox("Season", ["Select Season"] + season_list)

    with col3:
        state = st.selectbox("State", ["Select State"] + state_list)

    with col1:
        crop_year = st.text_input("Crop Year (YYYY)", "")

    with col2:
        area = st.text_input("Area (hectares)", "")

    with col3:
        production = st.text_input("Production (tonnes)", "")

    with col1:
        rainfall = st.text_input("Annual Rainfall (mm)", "")

    with col2:
        fertilizer = st.text_input("Fertilizer Usage (kg)", "")

    with col3:
        pesticide = st.text_input("Pesticide Usage (kg)", "")

    # Let user choose decimal precision
    decimals = st.slider("Decimal places", min_value=2, max_value=6, value=4)

    # --- Prediction ---
    yield_result = ""
    if st.button("📊 Predict Yield"):
        if crop == "Select Crop" or season == "Select Season" or state == "Select State":
            st.error("⚠️ Please select a valid Crop, Season, and State.")
        elif not all([crop_year, area, production, rainfall, fertilizer, pesticide]):
            st.error("⚠️ Please fill in all the fields before predicting.")
        else:
            try:
                # --- Encode using fixed mappings ---
                crop_enc   = crop_mapping[crop]
                season_enc = season_mapping[season]
                state_enc  = state_mapping[state]

                # Prepare user input in exact training order
                user_input = [
                    crop_enc, season_enc, state_enc,
                    int(crop_year), float(area), float(production),
                    float(rainfall), float(fertilizer), float(pesticide)
                ]

                # Scale input (same scaler used during training)
                user_input_scaled = scaler_yp.transform([user_input])

                # Run prediction
                y_pred = float(yield_prediction_model.predict(user_input_scaled)[0])

                # Prevent negative yields
                clipped = False
                if y_pred < 0:
                    y_pred = 0.0
                    clipped = True

                # Show with chosen precision
                y_str = f"{y_pred:.{decimals}f}"
                yield_result = f"🌾 Estimated Yield: **{y_str} t/ha**"
                if clipped:
                    st.info("Note: Model predicted a negative value; it was clipped to 0.0 t/ha.")

            except KeyError as e:
                yield_result = f"⚠️ Unknown label selected: {e}. Check for spacing/spelling."
            except Exception as e:
                yield_result = f"⚠️ Error: {str(e)}"

    st.success(yield_result)



# =============================
# Price Prediction Page
# =============================
elif selected == "Price Prediction":
    st.title("💰 Price Prediction using ML")
    st.write("Fill in the crop and environmental details to estimate market price.")

    # --- Input fields ---
    col1, col2, col3 = st.columns(3)

    # Crops (9)
    with col1:
        crop = st.selectbox("Crop", 
                            ["-- Select Crop --", 'Cotton(lint)', 'Onion', 'Potato', 'Rice', 
                             'Sugarcane', 'Wheat', 'Groundnut', 'Ragi', 'Banana'])

    # Season (6)
    with col2:
        season = st.selectbox("Season", 
                              ["-- Select Season --", 'Kharif', 'Whole Year', 'Autumn', 'Summer', 'Winter', 'Rabi'])

    # State (30)
    with col3:
        state = st.selectbox("State", 
                             ["-- Select State --", 'Assam', 'Karnataka', 'Meghalaya', 'West Bengal', 'Puducherry',
                              'Goa', 'Kerala', 'Andhra Pradesh', 'Tamil Nadu', 'Odisha',
                              'Bihar', 'Gujarat', 'Madhya Pradesh', 'Maharashtra', 'Mizoram',
                              'Punjab', 'Uttar Pradesh', 'Haryana', 'Himachal Pradesh', 'Tripura',
                              'Nagaland', 'Chhattisgarh', 'Uttarakhand', 'Jharkhand', 'Delhi',
                              'Manipur', 'Jammu and Kashmir', 'Telangana', 'Arunachal Pradesh', 'Sikkim'])

    # Area (numeric)
    with col1:
        area = st.number_input("Cultivated Area (hectares)", min_value=0.0, step=0.1)

    # Soil Type (encoded: 0–3)
    with col2:
        soil_type = st.selectbox("Soil Type", 
                                 ["-- Select Soil Type --", 'Alluvial Soil', 'Black Soil', 'Laterile Soil', 'Red/Yellow Soil'])

    # Pesticide Usage (2: low, high)
    with col3:
        pesticide_usage = st.selectbox("Pesticide Usage", ["-- Select Pesticide Usage --", 'low', 'high'])

    # pH (3: Average, Low, High)
    with col1:
        ph = st.selectbox("Soil pH Level", ["-- Select pH Level --", 'Average', 'Low', 'High'])

    # Temperature (3: Medium, Low, High)
    with col2:
        temperature = st.selectbox("Temperature", ["-- Select Temperature --", 'Medium', 'Low', 'High'])

    # Fertilizer Usage (3: High, Medium, Low)
    with col3:
        fertilizer_usage = st.selectbox("Fertilizer Usage", ["-- Select Fertilizer Usage --", 'High', 'Medium', 'Low'])

    # Rainfall (numeric)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, step=1.0)

    # --- Prediction ---
    price_result = ""

    if st.button("Predict Price"):
        # Validation check
        if (crop.startswith("--") or season.startswith("--") or state.startswith("--") or 
            soil_type.startswith("--") or pesticide_usage.startswith("--") or 
            ph.startswith("--") or temperature.startswith("--") or fertilizer_usage.startswith("--")):
            st.error("⚠️ Please select valid options for all fields before predicting.")
        else:
            try:
                # Collect user input
                user_input = [crop, season, state, area, soil_type, pesticide_usage, 
                              ph, temperature, fertilizer_usage, rainfall]

                # Run model prediction
                price_prediction = yield_prediction_model.predict([user_input])
                price_result = f"📈 Estimated Market Price: ₹{price_prediction[0]:,.2f} per quintal"

            except Exception as e:
                price_result = f"⚠️ Error: {str(e)}"

    st.success(price_result)

