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
# Load dataset (two header rows)
# -----------------------------
df = pd.read_excel("data/crop_data.xlsx", header=[0,1])

print("Dataset shape:", df.shape)
print(df.head())


results = []

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
# Extract crops automatically
# -----------------------------
crops = df.columns.get_level_values(0).unique()

print("Crops detected:", crops)


# -----------------------------
# Run models for each crop
# -----------------------------
for crop in crops:

    try:

        price = df[(crop, "Modal Price (₹)")]

        price = pd.to_numeric(price, errors="coerce")

        price = price.ffill()

        data = pd.DataFrame({"price": price})

        data["lag1"] = data["price"].shift(1)

        data = data.dropna()

        if len(data) < 20:
            continue


        X = data[["lag1"]]
        y = data["price"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )


        # Linear Regression
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        pred = lr.predict(X_test)
        evaluate("Linear Regression", crop, y_test, pred)


        # Random Forest
        rf = RandomForestRegressor()
        rf.fit(X_train, y_train)
        pred = rf.predict(X_test)
        evaluate("Random Forest", crop, y_test, pred)


        # SVR
        svr = SVR()
        svr.fit(X_train, y_train)
        pred = svr.predict(X_test)
        evaluate("SVR", crop, y_test, pred)


        # Decision Tree
        dt = DecisionTreeRegressor()
        dt.fit(X_train, y_train)
        pred = dt.predict(X_test)
        evaluate("Decision Tree", crop, y_test, pred)


        # XGBoost
        xgb = XGBRegressor()
        xgb.fit(X_train, y_train)
        pred = xgb.predict(X_test)
        evaluate("XGBoost", crop, y_test, pred)


        # ARIMA
        arima = ARIMA(y_train, order=(1,1,1))
        model = arima.fit()

        pred = model.forecast(len(y_test))

        evaluate("ARIMA", crop, y_test, pred)

    except:
        print("Skipping crop:", crop)


# -----------------------------
# Save results
# -----------------------------
os.makedirs("results", exist_ok=True)

results_df = pd.DataFrame(results)

results_df.to_csv(
    "results/model_comparison.csv",
    index=False
)

print("\nFinal Model Comparison")
print(results_df)
