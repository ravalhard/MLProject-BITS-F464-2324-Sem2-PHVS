import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor  # Import XGBRegressor

import numpy as np
import matplotlib.pyplot as plt

# Load the monthly average temperatures data
data = pd.read_csv('monthly_avg_temperatures.csv')

# Sort data by Year and Month if it's not sorted
data.sort_values(by=['Year', 'Month'], inplace=True)

# Generate lagged data for 24 months for AvgTemperature, Year, and Month
num_lags = 24
for i in range(1, num_lags + 1):
    data[f'Temp_lag_{i}'] = data['AvgTemperature'].shift(i)
    data[f'Year_lag_{i}'] = data['Year'].shift(i)
    data[f'Month_lag_{i}'] = data['Month'].shift(i)

# Drop rows with NaN values which are the first 24 rows
data.dropna(inplace=True)

# Define features (X) and target (y)
feature_columns = [f'Temp_lag_{i}' for i in range(1, num_lags + 1)] + \
                  [f'Year_lag_{i}' for i in range(1, num_lags + 1)] + \
                  [f'Month_lag_{i}' for i in range(1, num_lags + 1)]
X = data[feature_columns]
y = data['AvgTemperature']

# Split the data into train, validation, and test sets (60%, 20%, 20%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=0, shuffle=False)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=0, shuffle=False)

# Create and train the XGBoost model
model = XGBRegressor(n_estimators=300, random_state=0,  learning_rate=0.01 , max_depth = 3 , min_child_weight = 5)
model.fit(X_train, y_train)

# Predict on the validation set
y_pred_val = model.predict(X_val)

# Calculate and print the Mean Squared Error for the validation set
mse_val = mean_squared_error(y_val, y_pred_val)
print(f'Validation MSE: {mse_val}')

# Predict on the test set
y_pred_test = model.predict(X_test)

# Calculate and print the Mean Squared Error for the test set
mse_test = mean_squared_error(y_test, y_pred_test)
y_test = y_test.to_numpy()  # Assuming y_test is a pandas Series
print(f'Test MSE: {mse_test}')

# Calculate Mean Absolute Error (MAE)
mae_val = np.mean(np.abs(y_val - y_pred_val))
mae_test = np.mean(np.abs(y_test - y_pred_test))

# Calculate Root Mean Squared Error (RMSE)
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

# Calculate Mean Absolute Percentage Error (MAPE)
def calculate_mape(actual, predicted):
    return np.mean(np.abs((actual - predicted) / actual)) * 100

mape_val = calculate_mape(y_val, y_pred_val)
mape_test = calculate_mape(y_test, y_pred_test)

# Print errors
print(f'Validation MAE: {mae_val}')
print(f'Validation RMSE: {rmse_val}')
print(f'Validation MAPE: {mape_val}')

print(f'Test MAE: {mae_test}')
print(f'Test RMSE: {rmse_test}')
print(f'Test MAPE: {mape_test}')
