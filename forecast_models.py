import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA


# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_excel("data/crop_data.xlsx")

print("Original dataset shape:", df.shape)
print(df.head())


# -----------------------------
# Data Cleaning
# -----------------------------

# Convert all columns to numeric where possible
df = df.apply(pd.to_numeric, errors="coerce")

# Forward fill missing values
df = df.ffill()

# Drop completely empty rows
df = df.dropna(how="all")

print("After cleaning:", df.shape)

# Select numeric columns
df_numeric = df.select_dtypes(include=[np.number])

if df_numeric.shape[0] == 0:
    raise ValueError("Dataset has no usable rows")

# Select first numeric column as target
target = df_numeric.columns[0]

print("Target column:", target)


# -----------------------------
# Feature Engineering
# -----------------------------

# Lag feature (previous value)
df_numeric["lag1"] = df_numeric[target].shift(1)

# Remove NA values
df_numeric = df_numeric.dropna()

print("After lag feature:", df_numeric.shape)

# Features and Target
X = df_numeric[["lag1"]]
y = df_numeric[target]


# -----------------------------
# Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)


# -----------------------------
# Evaluation Function
# -----------------------------
results = []

def evaluate(name, y_true, y_pred):

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse
    })


# -----------------------------
# 1 Linear Regression
# -----------------------------
lr = LinearRegression()
lr.fit(X_train, y_train)
pred = lr.predict(X_test)
evaluate("Linear Regression", y_test, pred)


# -----------------------------
# 2 Random Forest
# -----------------------------
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
pred = rf.predict(X_test)
evaluate("Random Forest", y_test, pred)


# -----------------------------
# 3 Support Vector Regression
# -----------------------------
svr = SVR()
svr.fit(X_train, y_train)
pred = svr.predict(X_test)
evaluate("SVR", y_test, pred)


# -----------------------------
# 4 Decision Tree
# -----------------------------
dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)
pred = dt.predict(X_test)
evaluate("Decision Tree", y_test, pred)


# -----------------------------
# 5 XGBoost
# -----------------------------
xgb = XGBRegressor()
xgb.fit(X_train, y_train)
pred = xgb.predict(X_test)
evaluate("XGBoost", y_test, pred)


# -----------------------------
# 6 ARIMA (Time Series)
# -----------------------------
arima = ARIMA(y_train, order=(1,1,1))
model = arima.fit()

pred = model.forecast(len(y_test))
evaluate("ARIMA", y_test, pred)


# -----------------------------
# Save Results
# -----------------------------
os.makedirs("results", exist_ok=True)

results_df = pd.DataFrame(results)

results_df.to_csv(
    "results/model_comparison.csv",
    index=False
)

print("\nModel Comparison:")
print(results_df)
