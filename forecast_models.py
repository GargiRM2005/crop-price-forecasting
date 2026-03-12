importimport pandas as pd
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

# Forward fill missing values
df = df.ffill()

# Keep numeric columns only
df_numeric = df.select_dtypes(include=[np.number])

print("Numeric columns:", df_numeric.columns)

results = []


# -----------------------------
# Evaluation function
# -----------------------------
def evaluate(model_name, crop, y_true, y_pred):

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    results.append({
        "Crop": crop,
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse
    })


# -----------------------------
# Run models for each crop
# -----------------------------
for crop in df_numeric.columns:

    print("\nProcessing:", crop)

    data = df_numeric[[crop]].copy()

    # Skip if column mostly empty
    if data[crop].count() < 20:
        print("Skipping (too many missing values)")
        continue

    # Create lag feature
    data["lag1"] = data[crop].shift(1)

    data = data.dropna()

    if len(data) < 10:
        print("Skipping (not enough data)")
        continue

    X = data[["lag1"]]
    y = data[crop]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )


    # 1 Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    pred = lr.predict(X_test)
    evaluate("Linear Regression", crop, y_test, pred)


    # 2 Random Forest
    rf = RandomForestRegressor()
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)
    evaluate("Random Forest", crop, y_test, pred)


    # 3 SVR
    svr = SVR()
    svr.fit(X_train, y_train)
    pred = svr.predict(X_test)
    evaluate("SVR", crop, y_test, pred)


    # 4 Decision Tree
    dt = DecisionTreeRegressor()
    dt.fit(X_train, y_train)
    pred = dt.predict(X_test)
    evaluate("Decision Tree", crop, y_test, pred)


    # 5 XGBoost
    xgb = XGBRegressor()
    xgb.fit(X_train, y_train)
    pred = xgb.predict(X_test)
    evaluate("XGBoost", crop, y_test, pred)


    # 6 ARIMA
    try:
        arima = ARIMA(y_train, order=(1,1,1))
        model = arima.fit()

        pred = model.forecast(len(y_test))

        evaluate("ARIMA", crop, y_test, pred)

    except:
        print("ARIMA failed for", crop)


# -----------------------------
# Save Results
# -----------------------------
os.makedirs("results", exist_ok=True)

results_df = pd.DataFrame(results)

results_df.to_csv(
    "results/model_comparison.csv",
    index=False
)

print("\nFinal Model Comparison")
print(results_df)
