import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="PropVal ML • Intelligent House Price Estimator",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Modern UI / UX Styling
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1250px;
    }

    /* Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #312e81 100%);
        border-radius: 20px;
        padding: 2.2rem 2.5rem;
        color: #ffffff;
        margin-bottom: 1.8rem;
        box-shadow: 0 10px 30px -10px rgba(15, 23, 42, 0.4);
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .hero-badge-wrap {
        display: flex;
        gap: 8px;
        margin-bottom: 0.8rem;
        flex-wrap: wrap;
    }
    .hero-badge {
        background: rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(8px);
        color: #e0e7ff;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 999px;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0 0 0.5rem 0;
        line-height: 1.2;
        background: linear-gradient(to right, #ffffff, #c7d2fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.02rem;
        margin: 0;
        max-width: 720px;
        line-height: 1.5;
    }

    /* Card styling */
    .ui-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.4rem;
        box-shadow: 0 4px 15px -3px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.2rem;
    }

    /* Price Result Box */
    .result-card {
        background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 50%, #06b6d4 100%);
        border-radius: 18px;
        padding: 1.8rem;
        color: white;
        text-align: center;
        box-shadow: 0 12px 35px -8px rgba(79, 70, 229, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.25);
        margin-bottom: 1.2rem;
        transition: transform 0.2s ease;
    }
    .result-card:hover {
        transform: translateY(-2px);
    }
    .result-label {
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #e0e7ff;
        margin-bottom: 0.4rem;
    }
    .result-price-main {
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin: 0;
        line-height: 1.1;
    }
    .result-price-sub {
        font-size: 1.05rem;
        color: #bae6fd;
        margin-top: 0.4rem;
        font-weight: 500;
    }

    /* Stat pill */
    .stat-pill-container {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 10px;
        margin-top: 1rem;
    }
    .stat-pill {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        padding: 10px;
        border-radius: 12px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    .stat-pill-title {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #e0e7ff;
        margin-bottom: 2px;
    }
    .stat-pill-value {
        font-size: 1.05rem;
        font-weight: 700;
        color: #ffffff;
    }

    /* Sidebar Model Card */
    .sidebar-info-card {
        background: rgba(99, 102, 241, 0.08);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 12px;
        padding: 1rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    /* Custom Streamlit Metric Adjustments */
    [data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 12px 16px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.02);
    }
    [data-testid="stMetricLabel"] {
        font-weight: 600;
        font-size: 0.82rem;
        color: #64748b;
    }
    [data-testid="stMetricValue"] {
        font-weight: 700;
        font-size: 1.45rem;
        color: #0f172a;
    }

    /* Custom Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 8px 18px;
        font-weight: 600;
    }

    /* Section Headers */
    .section-title {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 1.15rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Cached Assets & Data
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("linear_regression_model.pkl")

@st.cache_data
def load_dataset():
    return pd.read_csv("sample_house_data.csv")

try:
    model = load_model()
    df = load_dataset()
    has_model = True
except Exception as e:
    st.error(f"Error loading model or dataset: {e}")
    st.stop()

# ---------------------------------------------------------
# Preset Scenarios Logic
# ---------------------------------------------------------
PRESETS = {
    "Custom": {"area": 1650, "beds": 3, "baths": 2, "age": 6, "distance": 8.0},
    "🏙️ Urban Compact Flat": {"area": 750, "beds": 1, "baths": 1, "age": 4, "distance": 3.0},
    "🏡 Suburban Family Home": {"area": 1850, "beds": 3, "baths": 2, "age": 8, "distance": 12.0},
    "🏰 Luxury Villa Estate": {"area": 3800, "beds": 5, "baths": 4, "age": 2, "distance": 5.5},
    "🏘️ Affordable Starter House": {"area": 1100, "beds": 2, "baths": 1, "age": 22, "distance": 18.0},
}

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1600585154340-be6161a56a0c?auto=format&fit=crop&w=600&q=80", use_container_width=True)
    st.markdown("### 🎛️ Scenario Presets")
    selected_preset = st.selectbox(
        "Choose a preconfigured property:",
        options=list(PRESETS.keys()),
        index=0,
        help="Quickly populate inputs with realistic property archetypes."
    )

    preset_vals = PRESETS[selected_preset]

    st.markdown("---")
    st.markdown("### 🧠 Model Performance")
    
    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        st.metric("R² Score", "0.984", help="Coefficient of determination: explains 98.4% of price variance.")
    with col_sb2:
        st.metric("MAE", "±10.28 L", help="Mean Absolute Error: average accuracy within ~10.3 Lakhs.")

    st.markdown("""
    <div class="sidebar-info-card">
        <span style="font-size: 0.85rem; font-weight: 700; color: #4338ca;">⚡ Algorithm Architecture</span><br/>
        <span style="font-size: 0.8rem; color: #475569;">
            Ordinary Least Squares (OLS) Linear Regression trained on 1,500 validated property records.
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Developed with ❤️ using Streamlit & Scikit-Learn.")

# ---------------------------------------------------------
# Hero Banner
# ---------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-badge-wrap">
        <span class="hero-badge">AI Valuations</span>
        <span class="hero-badge">Scikit-Learn ML</span>
        <span class="hero-badge">Interactive Dashboard</span>
    </div>
    <h1 class="hero-title">Real Estate Valuation Engine</h1>
    <p class="hero-subtitle">
        Accurately estimate property market worth using multivariate linear regression.
        Adjust architectural and locational parameters to observe instant mathematical price valuation.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Main App Tabs
# ---------------------------------------------------------
tab_estimator, tab_analytics, tab_explain, tab_data = st.tabs([
    "🏡 Live Valuation Engine",
    "📊 Market Distribution & Analytics",
    "📐 Mathematical Explainability",
    "📁 Dataset Explorer"
])

# ---------------------------------------------------------
# TAB 1: Live Estimator
# ---------------------------------------------------------
with tab_estimator:
    col_inputs, col_results = st.columns([1.15, 1], gap="large")

    with col_inputs:
        st.markdown('<div class="section-title">📐 Property Dimensions & Specifications</div>', unsafe_allow_html=True)
        
        area = st.slider(
            "Built-up Carpet Area (sq. ft.)",
            min_value=400,
            max_value=5000,
            value=preset_vals["area"],
            step=50,
            help="Total living and built-up floor area."
        )

        c_spec1, c_spec2 = st.columns(2)
        with c_spec1:
            bedrooms = st.number_input(
                "Bedrooms",
                min_value=1,
                max_value=8,
                value=preset_vals["beds"],
                step=1,
                help="Total number of bedrooms."
            )
        with c_spec2:
            bathrooms = st.number_input(
                "Bathrooms",
                min_value=1,
                max_value=6,
                value=preset_vals["baths"],
                step=1,
                help="Total number of functional bathrooms."
            )

        st.markdown('<div class="section-title" style="margin-top: 1.2rem;">📍 Age & Locational Proximity</div>', unsafe_allow_html=True)
        
        c_loc1, c_loc2 = st.columns(2)
        with c_loc1:
            age = st.slider(
                "Property Age (Years)",
                min_value=0,
                max_value=50,
                value=preset_vals["age"],
                step=1,
                help="Years elapsed since building construction."
            )
        with c_loc2:
            distance = st.slider(
                "Distance to City Centre (km)",
                min_value=0.0,
                max_value=40.0,
                value=preset_vals["distance"],
                step=0.5,
                help="Distance from the central business district."
            )

    # Prediction computation
    input_data = pd.DataFrame(
        [[area, bedrooms, bathrooms, age, distance]],
        columns=["area_sqft", "bedrooms", "bathrooms", "age_years", "distance_km"]
    )
    predicted_price_lakh = max(float(model.predict(input_data)[0]), 10.0)
    
    # Financial indicators
    price_in_rupees = predicted_price_lakh * 100000
    price_per_sqft = price_in_rupees / area
    
    # Currency formatting: Crores or Lakhs
    if predicted_price_lakh >= 100:
        cr_val = predicted_price_lakh / 100
        main_display = f"₹ {cr_val:.2f} Cr"
        sub_display = f"Equivalent to ₹{predicted_price_lakh:,.1f} Lakhs"
    else:
        main_display = f"₹ {predicted_price_lakh:,.1f} Lakhs"
        sub_display = f"Total Estimated Capital Value"

    # Estimated EMI (8.5% annual interest, 20 yrs tenure, 80% loan amount)
    loan_amount = price_in_rupees * 0.80
    r_monthly = (8.5 / 100) / 12
    n_months = 20 * 12
    emi_monthly = (loan_amount * r_monthly * (1 + r_monthly)**n_months) / ((1 + r_monthly)**n_months - 1)

    # Market percentile comparison
    percentile = (df['price_lakh'] < predicted_price_lakh).mean() * 100

    with col_results:
        # Beautiful Result Card
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">ML Estimated Market Valuation</div>
            <h1 class="result-price-main">{main_display}</h1>
            <div class="result-price-sub">{sub_display}</div>
            <div class="stat-pill-container">
                <div class="stat-pill">
                    <div class="stat-pill-title">Rate / Sq. Ft.</div>
                    <div class="stat-pill-value">₹{price_per_sqft:,.0f}</div>
                </div>
                <div class="stat-pill">
                    <div class="stat-pill-title">Est. Monthly EMI</div>
                    <div class="stat-pill-value">₹{emi_monthly:,.0f}</div>
                </div>
                <div class="stat-pill">
                    <div class="stat-pill-title">Market Rank</div>
                    <div class="stat-pill-value">Top {100 - percentile:.0f}%</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Feature Contribution Waterfall / Breakdown Chart
        st.markdown('<div class="section-title">📊 Feature Value Contributions</div>', unsafe_allow_html=True)
        
        coefs = model.coef_
        intercept = model.intercept_
        
        contrib_base = intercept
        contrib_area = area * coefs[0]
        contrib_beds = bedrooms * coefs[1]
        contrib_baths = bathrooms * coefs[2]
        contrib_age = age * coefs[3]
        contrib_dist = distance * coefs[4]

        contrib_df = pd.DataFrame({
            "Feature": ["Base Intercept", "Area Size", "Bedrooms", "Bathrooms", "Age Depreciation", "Distance Penalty"],
            "Contribution": [contrib_base, contrib_area, contrib_beds, contrib_baths, contrib_age, contrib_dist],
            "Type": ["Baseline", "Positive", "Positive", "Positive", "Negative", "Negative"]
        })
        
        fig_contrib = px.bar(
            contrib_df,
            x="Contribution",
            y="Feature",
            orientation="h",
            color="Type",
            color_discrete_map={"Baseline": "#64748b", "Positive": "#10b981", "Negative": "#ef4444"},
            text=contrib_df["Contribution"].apply(lambda v: f"{v:+.1f} L"),
            title="Impact of Each Attribute on Final Price (Lakhs)"
        )
        fig_contrib.update_layout(
            margin=dict(l=0, r=10, t=30, b=0),
            height=260,
            showlegend=False,
            xaxis_title="Price Impact (₹ Lakhs)",
            yaxis_title="",
            template="plotly_white"
        )
        st.plotly_chart(fig_contrib, use_container_width=True)

# ---------------------------------------------------------
# TAB 2: Market Distribution & Visual Analytics
# ---------------------------------------------------------
with tab_analytics:
    st.markdown('<div class="section-title">📈 Benchmark vs. Market Dataset (1,500 Properties)</div>', unsafe_allow_html=True)
    
    col_viz1, col_viz2 = st.columns([1, 1], gap="medium")
    
    with col_viz1:
        # Area vs Price Scatter with current property highlighted
        fig_scatter = px.scatter(
            df,
            x="area_sqft",
            y="price_lakh",
            color="distance_km",
            color_continuous_scale="Viridis_r",
            opacity=0.45,
            labels={"area_sqft": "Area (sq. ft.)", "price_lakh": "Price (₹ Lakh)", "distance_km": "Distance (km)"},
            title="Market Trend: Area vs Price (Color = Distance km)"
        )
        # Add highlighted marker for user property
        fig_scatter.add_trace(go.Scatter(
            x=[area],
            y=[predicted_price_lakh],
            mode="markers+text",
            marker=dict(symbol="star", size=20, color="#ef4444", line=dict(width=2, color="white")),
            text=["📍 Your Property"],
            textposition="top center",
            name="Your Property"
        ))
        fig_scatter.update_layout(height=400, template="plotly_white", margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_viz2:
        # Distance vs Price with trendline
        fig_dist = px.scatter(
            df,
            x="distance_km",
            y="price_lakh",
            color="bedrooms",
            trendline="ols",
            opacity=0.45,
            labels={"distance_km": "Distance to Center (km)", "price_lakh": "Price (₹ Lakh)", "bedrooms": "Beds"},
            title="Distance Decay: Price Depreciation by City Distance"
        )
        fig_dist.add_trace(go.Scatter(
            x=[distance],
            y=[predicted_price_lakh],
            mode="markers+text",
            marker=dict(symbol="diamond", size=16, color="#e11d48", line=dict(width=2, color="white")),
            text=["📍 Here"],
            textposition="top right",
            name="Your Property"
        ))
        fig_dist.update_layout(height=400, template="plotly_white", margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_dist, use_container_width=True)

    # Distribution histogram
    fig_hist = px.histogram(
        df,
        x="price_lakh",
        nbins=40,
        marginal="box",
        color_discrete_sequence=["#6366f1"],
        title="Overall Market Price Distribution & Range Spread"
    )
    fig_hist.add_vline(x=predicted_price_lakh, line_width=3, line_dash="dash", line_color="#ef4444", annotation_text=f"Your Estimate: ₹{predicted_price_lakh:.1f}L")
    fig_hist.update_layout(height=320, template="plotly_white", margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_hist, use_container_width=True)

# ---------------------------------------------------------
# TAB 3: Mathematical Explainability
# ---------------------------------------------------------
with tab_explain:
    st.markdown('<div class="section-title">📐 Linear Regression Mathematical Formulation</div>', unsafe_allow_html=True)
    st.markdown("""
    The fitted linear regression model calculates property valuation through an additive linear combination of features:
    """)
    
    st.latex(r"""
    \text{Price (Lakh)} = \beta_0 + \beta_1(\text{Area}) + \beta_2(\text{Bedrooms}) + \beta_3(\text{Bathrooms}) + \beta_4(\text{Age}) + \beta_5(\text{Distance})
    """)

    st.markdown("#### Dynamic Parameter Valuation Equation:")
    st.latex(f"""
    \\text{{Price}} = {intercept:.2f} + ({coefs[0]:.4f} \\times {area}) + ({coefs[1]:.2f} \\times {bedrooms}) + ({coefs[2]:.2f} \\times {bathrooms}) + ({coefs[3]:.2f} \\times {age}) + ({coefs[4]:.2f} \\times {distance:.1f})
    """)

    st.markdown("---")
    st.markdown("#### Marginal Feature Weights (Model Coefficients):")
    
    coeff_data = pd.DataFrame({
        "Feature Attribute": [
            "Baseline Intercept (β₀)",
            "Area (sq. ft.) (β₁)",
            "Bedrooms (β₂)",
            "Bathrooms (β₃)",
            "Age (years) (β₄)",
            "Distance (km) (β₅)"
        ],
        "Coefficient Value": [
            f"{intercept:+.3f}",
            f"{coefs[0]:+.4f}",
            f"{coefs[1]:+.3f}",
            f"{coefs[2]:+.3f}",
            f"{coefs[3]:+.3f}",
            f"{coefs[4]:+.3f}"
        ],
        "Practical Economic Interpretation": [
            "Theoretical starting price before feature adjustments",
            f"Every 100 sq.ft added increases property value by ₹{coefs[0]*100:.2f} Lakh",
            f"Each additional bedroom adds approximately ₹{coefs[1]:.2f} Lakh",
            f"Each additional bathroom adds approximately ₹{coefs[2]:.2f} Lakh",
            f"Every year of property age depreciates value by ₹{abs(coefs[3]):.2f} Lakh",
            f"Every 1 km further away from city center reduces price by ₹{abs(coefs[4]):.2f} Lakh"
        ]
    })
    st.dataframe(coeff_data, hide_index=True, use_container_width=True)

# ---------------------------------------------------------
# TAB 4: Dataset Explorer
# ---------------------------------------------------------
with tab_data:
    st.markdown('<div class="section-title">📁 Sample Housing Dataset Explorer</div>', unsafe_allow_html=True)
    
    c_m1, c_m2, c_m3, c_m4 = st.columns(4)
    with c_m1:
        st.metric("Total Records", f"{len(df):,}")
    with c_m2:
        st.metric("Avg Price", f"₹ {df['price_lakh'].mean():.1f} L")
    with c_m3:
        st.metric("Median Area", f"{df['area_sqft'].median():,.0f} sq.ft")
    with c_m4:
        st.metric("Avg Distance", f"{df['distance_km'].mean():.1f} km")

    st.markdown("##### Filtered Data Records:")
    max_price_filter = st.slider("Filter by Maximum Price (₹ Lakh)", int(df.price_lakh.min()), int(df.price_lakh.max()), int(df.price_lakh.max()))
    filtered_df = df[df.price_lakh <= max_price_filter]
    
    st.dataframe(
        filtered_df.head(100),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        label="📥 Download Dataset (CSV)",
        data=df.to_csv(index=False),
        file_name="housing_dataset.csv",
        mime="text/csv"
    )

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center; color: #64748b; font-size: 0.82rem;">
    <span>⚡ Powered by <strong>Streamlit</strong> & <strong>Scikit-Learn Linear Regression</strong></span>
    <span>💡 Educational & Portfolio Showcase</span>
</div>
""", unsafe_allow_html=True)
