# 🏡 PropVal ML • House Price Prediction App

An interactive, modern real-estate valuation web application built with **Streamlit**, **Scikit-Learn**, and **Plotly**.

---

## ✨ Features & Highlights

- **🎨 Modern Glassmorphism UI/UX**: Designed with custom CSS, typography, responsive cards, and dynamic visual badges.
- **⚡ Real-Time Price Estimation**: Instant multivariate valuation based on area, bedrooms, bathrooms, property age, and distance to city center.
- **🎛️ 1-Click Scenario Presets**: Quickly preview real-estate archetypes (Urban Compact, Suburban Family, Luxury Villa, Starter House).
- **📊 Feature Breakdown & Waterfall Impact**: Visualize the exact positive and negative contributions of each property feature to the final price.
- **📈 Interactive Plotly Analytics**: Compare your estimated home against 1,500 market data points with 2D & 3D scatter plots and distribution curves.
- **📐 Mathematical Explainability**: Dynamic equation rendering (LaTeX) showing fitted weights and marginal costs.
- **💳 Financial Metrics**: Displays total price in Lakhs/Crores, rate per square foot, and estimated monthly home loan EMI.

---

## 🛠️ Tech Stack

- **Frontend & App Framework**: [Streamlit](https://streamlit.io/)
- **Machine Learning**: [Scikit-Learn](https://scikit-learn.org/) (Linear Regression, $R^2 \approx 0.984$)
- **Interactive Visualizations**: [Plotly](https://plotly.com/python/)
- **Data Manipulation**: Pandas & NumPy

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Akshay-Notfound/Linear_Regression_Streamlit.git
cd Linear_Regression_Streamlit
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit application
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 🔁 Retraining the Model (Optional)

To regenerate sample data and retrain the Linear Regression model:
```bash
python train_model.py
```
This updates `linear_regression_model.pkl` and `sample_house_data.csv`.
