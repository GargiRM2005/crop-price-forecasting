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
# Load dataset with 2 headers
# -----------------------------
df = pd.read_excel("data/crop_data.xlsx", header=[0,1])

print("Dataset shape:", df.shape)


results = []
monthly_predictions = []


# -----------------------------
# Error metrics
# -----------------------------
def calculate_metrics(y_true, y_pred):

    mae = mean_absolute_error(y_true, y_pred)

    mse = mean_squared_error(y_true, y_pred)

    rmse = np.sqrt(mse)

    mad = np.mean(np.abs(y_true - y_pred))

    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return mae, mse, rmse, mad, mape


# -----------------------------
# Detect crops automatically
# -----------------------------
crops = df.columns.get_level_values(0).unique()

print("Detected crops:", crops)


# -----------------------------
# Run forecasting for each crop
# -----------------------------
for crop in crops:

    try:

        print("\nProcessing:", crop)

        price = df[(crop, "Modal Price (₹)")]

        price = pd.to_numeric(price, errors="coerce")

        price = price.ffill()

        data = pd.DataFrame({
            "price": price
        })

        data["lag1"] = data["price"].shift(1)

        data = data.dropna()

        if len(data) < 20:
            print("Skipping small dataset")
            continue


        X = data[["lag1"]]
        y = data["price"]

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


        # -----------------------------
        # ML models
        # -----------------------------
        for name, model in models.items():

            model.fit(X_train, y_train)

            pred = model.predict(X_test)

            mae, mse, rmse, mad, mape = calculate_metrics(y_test, pred)

            results.append({

                "Crop": crop,
                "Model": name,
                "MAE": mae,
                "MSE": mse,
                "RMSE": rmse,
                "MAD": mad,
                "MAPE (%)": mape

            })

            for i in range(len(pred)):

                monthly_predictions.append({

                    "Crop": crop,
                    "Model": name,
                    "Month_Index": i,
                    "Actual": y_test.iloc[i],
                    "Predicted": pred[i]

                })


        # -----------------------------
        # ARIMA
        # -----------------------------
        arima = ARIMA(y_train, order=(1,1,1))

        model = arima.fit()

        pred = model.forecast(len(y_test))

        mae, mse, rmse, mad, mape = calculate_metrics(y_test, pred)

        results.append({

            "Crop": crop,
            "Model": "ARIMA",
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "MAD": mad,
            "MAPE (%)": mape

        })

        for i in range(len(pred)):

            monthly_predictions.append({

                "Crop": crop,
                "Model": "ARIMA",
                "Month_Index": i,
                "Actual": y_test.iloc[i],
                "Predicted": pred.iloc[i]

            })


    except Exception as e:

        print("Skipping", crop, "Error:", e)



# -----------------------------
# Save results
# -----------------------------
os.makedirs("results", exist_ok=True)

results_df = pd.DataFrame(results)

monthly_df = pd.DataFrame(monthly_predictions)


results_df.to_csv(
    "results/model_comparison.csv",
    index=False
)

monthly_df.to_csv(
    "results/monthly_predictions.csv",
    index=False
)


print("\nModel Comparison Table")
print(results_df)

print("\nMonthly Predictions")
print(monthly_df.head())
