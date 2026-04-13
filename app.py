import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UrbanNest Analytics – Rent Predictor",
    page_icon="🏠",
    layout="centered",
)

st.title("🏠 UrbanNest Analytics")
st.subheader("Dynamic House Rent Prediction Engine")
st.markdown("Fill in the property details below and click **Predict** to get the estimated monthly rent.")
st.divider()

# ── Load model & encoders ─────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open("models/best_rf_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    with open("models/feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)
    return model, encoders, feature_names

try:
    model, encoders, feature_names = load_artifacts()
except FileNotFoundError:
    st.error("⚠️ Model files not found. Please run `train.ipynb` first to generate `models/best_rf_model.pkl`, `models/encoders.pkl`, and `models/feature_names.pkl`.")
    st.stop()

# ── Location data per city ────────────────────────────────────────────────────
CITY_LOCATIONS = {
    "Mumbai": sorted(encoders["location"].classes_[
        [i for i, loc in enumerate(encoders["location"].classes_)
         if loc in ["Thane West","Andheri East","Airoli","Borivali West","Powai",
                    "Kandivali West","Malad West","Goregaon West","Bandra West",
                    "Chembur","Kurla","Ghatkopar","Mulund West","Vikhroli"]]
    ].tolist() + list(encoders["location"].classes_)).tolist()),
    "Delhi": [],
    "Pune": [],
    "Hisar": [],
}

# Fallback: just use all encoded location labels
ALL_LOCATIONS = sorted(encoders["location"].classes_.tolist())

# ── City → approximate lat/lon centers ───────────────────────────────────────
CITY_COORDS = {
    "Mumbai": (19.1293, 72.8847),
    "Delhi":  (28.5718, 77.1965),
    "Pune":   (18.5754, 73.8953),
    "Hisar":  (29.1492, 75.7217),
}

# ── Input form ────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    city = st.selectbox("🌆 City", options=["Mumbai", "Delhi", "Pune", "Hisar"])

with col2:
    location = st.selectbox("📍 Location / Neighbourhood", options=ALL_LOCATIONS)

col3, col4 = st.columns(2)

with col3:
    property_type = st.selectbox(
        "🏢 Property Type",
        options=["Apartment", "Studio Apartment", "Independent Floor",
                 "Independent House", "Villa", "penthouse"],
    )

with col4:
    status = st.selectbox(
        "🛋️ Furnishing Status",
        options=["Unfurnished", "Semi-Furnished", "Furnished"],
    )

st.markdown("#### Property Specifications")
col5, col6, col7 = st.columns(3)

with col5:
    size = st.number_input("📐 Size (ft²)", min_value=150, max_value=15000, value=800, step=50)

with col6:
    rooms_num = st.number_input("🚪 Number of Rooms", min_value=1, max_value=12, value=2)

with col7:
    num_bathrooms = st.number_input("🚿 Bathrooms", min_value=0, max_value=10, value=1)

col8, col9, col10 = st.columns(3)

with col8:
    num_balconies = st.number_input("🏗️ Balconies", min_value=0, max_value=8, value=1)

with col9:
    bhk = st.selectbox("🛏️ Type", options=["BHK (Bedroom-Hall-Kitchen)", "RK (Room-Kitchen)"])
    bhk_val = 1 if bhk.startswith("BHK") else 0

with col10:
    is_negotiable = st.selectbox("💬 Negotiable?", options=["No", "Yes"])
    is_negotiable_val = 1 if is_negotiable == "Yes" else 0

st.markdown("#### Financial & Location Details")
col11, col12 = st.columns(2)

with col11:
    security_deposit = st.number_input(
        "🔒 Security Deposit (INR)", min_value=0, max_value=5_000_000,
        value=50_000, step=5000,
    )

with col12:
    verification_days = st.number_input(
        "📅 Days Since Posting", min_value=0, max_value=1825,
        value=30, step=1,
        help="How many days ago was this property posted/verified?",
    )

# Lat/lon: auto-set from city centre; allow user to override
default_lat, default_lon = CITY_COORDS.get(city, (20.0, 78.0))
col13, col14 = st.columns(2)
with col13:
    latitude = st.number_input("🌐 Latitude", value=default_lat, format="%.6f")
with col14:
    longitude = st.number_input("🌐 Longitude", value=default_lon, format="%.6f")

st.divider()

# ── Prediction ────────────────────────────────────────────────────────────────
if st.button("🔍 Predict Rent", use_container_width=True, type="primary"):

    # Build raw input dict
    raw_input = {
        "location"        : location,
        "city"            : city,
        "latitude"        : latitude,
        "longitude"       : longitude,
        "numBathrooms"    : num_bathrooms,
        "numBalconies"    : num_balconies,
        "isNegotiable"    : is_negotiable_val,
        "SecurityDeposit" : security_deposit,
        "Status"          : status,
        "Size_ft²"        : size,
        "BHK"             : bhk_val,
        "rooms_num"       : rooms_num,
        "property_type"   : property_type,
        "verification_days": float(verification_days),
    }

    # Apply label encoders to categorical columns
    CATEGORICAL_COLS = ["location", "city", "Status", "property_type"]
    for col in CATEGORICAL_COLS:
        le = encoders[col]
        val = str(raw_input[col])
        if val in le.classes_:
            raw_input[col] = int(le.transform([val])[0])
        else:
            # Unseen label — use most frequent class (index 0)
            raw_input[col] = 0
            st.warning(f"'{val}' not seen during training for '{col}'. Using default encoding.")

    # Assemble feature vector in the correct order
    input_vector = np.array([[raw_input[feat] for feat in feature_names]])

    prediction = model.predict(input_vector)[0]

    st.success(f"### 🏷️ Estimated Monthly Rent: ₹{prediction:,.0f}")

    with st.expander("📊 Input summary"):
        display = {
            "City": city, "Location": location, "Property Type": property_type,
            "Furnishing": status, "Size (ft²)": size, "Rooms": rooms_num,
            "Bathrooms": num_bathrooms, "Balconies": num_balconies,
            "Type": bhk, "Negotiable": is_negotiable,
            "Security Deposit (INR)": f"₹{security_deposit:,}",
            "Days Since Posting": verification_days,
            "Latitude": latitude, "Longitude": longitude,
        }
        for k, v in display.items():
            st.markdown(f"**{k}:** {v}")

st.markdown(
    "<br><small style='color:grey'>UrbanNest Analytics — Rent Prediction Engine · Powered by Random Forest</small>",
    unsafe_allow_html=True,
)
