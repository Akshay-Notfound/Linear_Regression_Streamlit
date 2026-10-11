import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(42)
n = 1500
df = pd.DataFrame({
    "area_sqft": rng.integers(400, 5001, n),
    "bedrooms": rng.integers(1, 9, n),
    "bathrooms": rng.integers(1, 7, n),
    "age_years": rng.integers(0, 51, n),
    "distance_km": rng.uniform(0, 40, n).round(1),
})
df["price_lakh"] = (0.075*df.area_sqft + 8*df.bedrooms + 6*df.bathrooms - 1.1*df.age_years - 2.2*df.distance_km + rng.normal(0, 12, n)).clip(10)
X, y = df.drop(columns="price_lakh"), df.price_lakh
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)
pred = model.predict(X_test)
print(f"R2: {r2_score(y_test,pred):.3f} | MAE: {mean_absolute_error(y_test,pred):.2f} lakh")
joblib.dump(model, "linear_regression_model.pkl")
df.to_csv("sample_house_data.csv", index=False)
