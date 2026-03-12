import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA

# ---------------------------
# LOAD DATASET
# ---------------------------

file_path = "crop_data.xlsx"

df = pd.read_excel(file_path, header=1)

print("Original dataset shape:", df.shape)
print(df.head())

# Convert values to numeric where possible
df = df.apply(pd.to_numeric, errors="ignore")

# Fill missing values with previous value
df = df.ffill()

# Select numeric columns (these represent crops)
numeric_cols = df.select_dtypes(include=np.number).columns

print("Detected crop columns:", numeric_cols)

# Create results folder
os.makedirs("results", exist_ok=True)

all_results = []

# ---------------------------
# METRICS FUNCTION
# ---------------------------

def calculate_metrics(actual, predicted):

    mad = np.mean(np.abs(actual - predicted))

    mse = mean_squared_error(actual, predicted)

    rmse = np.sqrt(mse)

    mape = np.mean(np.abs((actual - predicted) / actual)) * 100

    error_percent = mape

    return mad, mse, rmse, mape, error_percent


# ---------------------------
# FORECAST LOOP
# ---------------------------

for crop in numeric_cols:

    print("\nProcessing crop:", crop)

    series = df[crop].dropna()

    # create lag feature
    lag = series.shift(1)

    data = pd.DataFrame({
        "y": series,
        "lag1": lag
    }).dropna()

    if len(data) < 10:
        print("Skipping crop (not enough data)")
        continue

    X = data[["lag1"]]
    y = data["y"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        shuffle=False
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(),
        "SVR": SVR(),
        "Decision Tree": DecisionTreeRegressor(),
        "XGBoost": XGBRegressor()
    }

    crop_results = []

    # ---------------------------
    # MACHINE LEARNING MODELS
    # ---------------------------

    for name, model in models.items():

        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        mad, mse, rmse, mape, error_percent = calculate_metrics(y_test, pred)

        crop_results.append([
            name,
            mad,
            mse,
            rmse,
            mape,
            error_percent
        ])

        all_results.append({
            "Crop": crop,
            "Model": name,
            "MAD": mad,
            "MSE": mse,
            "RMSE": rmse,
            "MAPE (%)": mape,
            "Error (%)": error_percent
        })

    # ---------------------------
    # ARIMA MODEL
    # ---------------------------

    try:

        arima = ARIMA(y_train, order=(1,1,1))
        model = arima.fit()

        pred = model.forecast(len(y_test))

        mad, mse, rmse, mape, error_percent = calculate_metrics(y_test, pred)

        crop_results.append([
            "ARIMA",
            mad,
            mse,
            rmse,
            mape,
            error_percent
        ])

        all_results.append({
            "Crop": crop,
            "Model": "ARIMA",
            "MAD": mad,
            "MSE": mse,
            "RMSE": rmse,
            "MAPE (%)": mape,
            "Error (%)": error_percent
        })

    except:
        print("ARIMA failed for", crop)

    # ---------------------------
    # CREATE TABLE FOR CROP
    # ---------------------------

    crop_table = pd.DataFrame(
        crop_results,
        columns=[
            "Model",
            "MAD",
            "MSE",
            "RMSE",
            "MAPE (%)",
            "Error (%)"
        ]
    )

    print("\nForecast Error Table for", crop)
    print(crop_table)

    # Save crop table
    crop_table.to_csv(f"results/{crop}_forecast_results.csv", index=False)


# ---------------------------
# SAVE ALL RESULTS
# ---------------------------

final_df = pd.DataFrame(all_results)

final_df.to_csv("results/model_comparison_all_crops.csv", index=False)

print("\nAll crop results saved to:")
print("results/model_comparison_all_crops.csv")
