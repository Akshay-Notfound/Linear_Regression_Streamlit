import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")
st.markdown("""<style>.stApp{background:linear-gradient(135deg,#f8fbff,#eef5ff)}
[data-testid='stMetric']{background:white;padding:18px;border-radius:14px;box-shadow:0 4px 18px #dbe7f5}
.result{padding:24px;border-radius:16px;background:#0b5ed7;color:white;text-align:center}</style>""", unsafe_allow_html=True)

model = joblib.load("linear_regression_model.pkl")
st.title("🏠 House Price Prediction")
st.caption("Linear Regression deployment using Streamlit")

left, right = st.columns([1, 1])
with left:
    st.subheader("Property details")
    area = st.slider("Area (sq. ft.)", 400, 5000, 1500, 50)
    bedrooms = st.number_input("Bedrooms", 1, 8, 3)
    bathrooms = st.number_input("Bathrooms", 1, 6, 2)
    age = st.slider("Property age (years)", 0, 50, 5)
    distance = st.slider("Distance from city centre (km)", 0.0, 40.0, 8.0, 0.5)
with right:
    st.subheader("Model information")
    c1, c2 = st.columns(2)
    c1.metric("R² score", "0.96")
    c2.metric("Algorithm", "Linear Regression")
    st.info("The model estimates price from area, rooms, property age, and location distance.")
    if st.button("Predict house price", type="primary", use_container_width=True):
        x = pd.DataFrame([[area, bedrooms, bathrooms, age, distance]], columns=["area_sqft","bedrooms","bathrooms","age_years","distance_km"])
        prediction = max(float(model.predict(x)[0]), 0)
        st.markdown(f"<div class='result'><h3>Estimated Price</h3><h1>₹{prediction:,.0f} lakh</h1></div>", unsafe_allow_html=True)
        st.write("Inputs used:", x)

st.divider()
st.caption("Educational demo • Predictions are illustrative and not financial advice.")
