import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA

import os
# Load dataset
df = pd.read_excel("data/crop_data.xlsx")

print("Original shape:", df.shape)
print(df.head())

# Keep numeric columns only
df_numeric = df.select_dtypes(include=[np.number])

if df_numeric.shape[1] == 0:
    raise ValueError("No numeric columns found in dataset")

# Choose first numeric column as target
target = df_numeric.columns[0]

# Fill missing values
df_numeric = df_numeric.ffill()

# Create lag feature
df_numeric["lag1"] = df_numeric[target].shift(1)

# Remove rows with NA
df_numeric = df_numeric.dropna()

print("Processed shape:", df_numeric.shape)

# Features and target
X = df_numeric[["lag1"]]
y = df_numeric[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

results = []

def evaluate(name, y_true, y_pred):

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse
    })


# 1 Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
pred = lr.predict(X_test)
evaluate("Linear Regression", y_test, pred)


# 2 Random Forest
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
pred = rf.predict(X_test)
evaluate("Random Forest", y_test, pred)


# 3 SVR
svr = SVR()
svr.fit(X_train, y_train)
pred = svr.predict(X_test)
evaluate("SVR", y_test, pred)


# 4 Decision Tree
dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)
pred = dt.predict(X_test)
evaluate("Decision Tree", y_test, pred)


# 5 XGBoost
xgb = XGBRegressor()
xgb.fit(X_train, y_train)
pred = xgb.predict(X_test)
evaluate("XGBoost", y_test, pred)


# 6 ARIMA
arima = ARIMA(y_train, order=(1,1,1))
model = arima.fit()

pred = model.forecast(len(y_test))
evaluate("ARIMA", y_test, pred)


os.makedirs("results", exist_ok=True)

results_df = pd.DataFrame(results)
results_df.to_csv("results/model_comparison.csv", index=False)

print(results_df)
