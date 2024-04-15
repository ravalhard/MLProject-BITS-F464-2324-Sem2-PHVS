import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('monthly_avg_temperatures.csv')
data.sort_values(by=['Year', 'Month'], inplace=True)

# Generate lagged data
num_lags = 24  # Number of lag months
for i in range(1, num_lags + 1):
    data[f'Temp_lag_{i}'] = data['AvgTemperature'].shift(i)

# Drop rows with NaN values
data.dropna(inplace=True)

# Define features and target
features = [f'Temp_lag_{i}' for i in range(1, num_lags + 1)]
X = data[features]
y = data['AvgTemperature']

# Rescale the features and the target
scaler_x = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler_x.fit_transform(X)

scaler_y = MinMaxScaler(feature_range=(0, 1))
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

# Reshape input to be [samples, time steps, features]
X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

# Split the data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_scaled, test_size=0.4, random_state=0, shuffle=False)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=0, shuffle=False)

# Create the LSTM model
model = Sequential([
    LSTM(40, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
    LSTM(20),
    Dense(1)
])
model.compile(optimizer='rmsprop', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=24, validation_data=(X_val, y_val), verbose=3)

# Predict on the validation set
y_pred_val = model.predict(X_val)

# Convert predictions back to original scale
y_pred_val_original = scaler_y.inverse_transform(y_pred_val)
y_val_original = scaler_y.inverse_transform(y_val)

# Calculate MSE for the validation set on original scale
mse_val = mean_squared_error(y_val_original, y_pred_val_original)
print(f'Validation MSE: {mse_val}')

# Predict on the test set
y_pred_test = model.predict(X_test)

# Convert test predictions back to original scale
y_pred_test_original = scaler_y.inverse_transform(y_pred_test)
y_test_original = scaler_y.inverse_transform(y_test)

# Calculate MSE for the test set on original scale
mse_test = mean_squared_error(y_test_original, y_pred_test_original)
print(f'Test MSE: {mse_test}')

# Optionally, plot the results
# Calculate Mean Absolute Error (MAE)
mae_val = np.mean(np.abs(y_val_original - y_pred_val_original))
mae_test = np.mean(np.abs(y_test_original - y_pred_test_original))

# Calculate Root Mean Squared Error (RMSE)
rmse_val = np.sqrt(mean_squared_error(y_val_original, y_pred_val_original))
rmse_test = np.sqrt(mean_squared_error(y_test_original, y_pred_test_original))

# Calculate Mean Absolute Percentage Error (MAPE)
def calculate_mape(actual, predicted):
    return np.mean(np.abs((actual - predicted) / actual)) * 100

mape_val = calculate_mape(y_val_original, y_pred_val_original)
mape_test = calculate_mape(y_test_original, y_pred_test_original)

# Print errors
print(f'Validation MAE: {mae_val}')
print(f'Validation RMSE: {rmse_val}')
print(f'Validation MAPE: {mape_val}')

print(f'Test MAE: {mae_test}')
print(f'Test RMSE: {rmse_test}')
print(f'Test MAPE: {mape_test}')


