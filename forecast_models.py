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

from reportlab.platypus import SimpleDocTemplate, Table

# ----------------------------
# LOAD DATASET
# ----------------------------

df = pd.read_excel("crop_data.xlsx", header=1)

print("Original dataset shape:", df.shape)
print(df.head())

# convert numbers
df = df.apply(pd.to_numeric, errors="ignore")

# fill missing values with previous value
df = df.ffill()

# select numeric columns
numeric_cols = df.select_dtypes(include=np.number).columns

print("Numeric columns:", numeric_cols)

results = []
monthly_predictions = []

# ----------------------------
# ERROR METRICS
# ----------------------------

def calculate_metrics(y_true, y_pred):

    mad = np.mean(np.abs(y_true - y_pred))

    mse = mean_squared_error(y_true, y_pred)

    rmse = np.sqrt(mse)

    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    error_percent = mape

    return mad, mse, rmse, mape, error_percent


# ----------------------------
# MODEL LOOP
# ----------------------------

for crop in numeric_cols[:5]:

    print("\nProcessing crop:", crop)

    series = df[crop].dropna()

    lag = series.shift(1)

    data = pd.DataFrame({
        "y": series,
        "lag1": lag
    }).dropna()

    if len(data) < 10:
        print("Skipping crop (too little data)")
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

    # ----------------------------
    # RUN ML MODELS
    # ----------------------------

    for name, model in models.items():

        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        mad, mse, rmse, mape, error_percent = calculate_metrics(y_test, pred)

        results.append({
            "Crop": crop,
            "Model": name,
            "MAD": mad,
            "MSE": mse,
            "RMSE": rmse,
            "MAPE (%)": mape,
            "Error (%)": error_percent
        })

        # save monthly predictions
        for i in range(len(pred)):

            monthly_predictions.append({
                "Crop": crop,
                "Model": name,
                "Actual": y_test.iloc[i],
                "Predicted": pred[i]
            })

    # ----------------------------
    # ARIMA MODEL
    # ----------------------------

    try:

        arima = ARIMA(y_train, order=(1,1,1))

        model = arima.fit()

        pred = model.forecast(len(y_test))

        mad, mse, rmse, mape, error_percent = calculate_metrics(y_test, pred)

        results.append({
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


# ----------------------------
# SAVE RESULTS
# ----------------------------

os.makedirs("results", exist_ok=True)

results_df = pd.DataFrame(results)
monthly_df = pd.DataFrame(monthly_predictions)

results_df.to_csv("results/model_comparison.csv", index=False)
monthly_df.to_csv("results/monthly_predictions.csv", index=False)

print("\nModel Comparison Table")
print(results_df)

# ----------------------------
# EXPORT PDF
# ----------------------------

data = [results_df.columns.tolist()] + results_df.values.tolist()

pdf = SimpleDocTemplate("results/model_results.pdf")

table = Table(data)

pdf.build([table])

print("\nFiles Generated:")
print("results/model_comparison.csv")
print("results/monthly_predictions.csv")
print("results/model_results.pdf")
