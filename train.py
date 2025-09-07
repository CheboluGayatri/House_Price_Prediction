import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import os
import warnings
import numpy as np

# ----------------------------
# Suppress warnings
# ----------------------------
warnings.filterwarnings("ignore")

# ----------------------------
# Load housing dataset
# ----------------------------
df = pd.read_csv("Housing.csv")
df = df.dropna()

# Convert categorical features to numeric (One-Hot Encoding)
df = pd.get_dummies(df, columns=[
    'mainroad', 'guestroom', 'basement',
    'hotwaterheating', 'airconditioning',
    'prefarea', 'furnishingstatus'
], drop_first=True)

# ----------------------------
# Features and Target
# ----------------------------
X = df.drop("price", axis=1)
y = df["price"]

# ----------------------------
# Split data (80% train, 20% test)
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# Train Model
# ----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ----------------------------
# Evaluate Model
# ----------------------------
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"📊 Model Performance:")
print(f"   R² Score : {r2:.4f}")
print(f"   RMSE     : {rmse:.2f}")

# ----------------------------
# Save Model and Features
# ----------------------------
os.makedirs("models", exist_ok=True)
joblib.dump((model, X.columns.tolist()), "models/house_price_model.joblib")

print("✅ Model trained and saved to models/house_price_model.joblib")
