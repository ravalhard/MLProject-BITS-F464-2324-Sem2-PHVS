import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# Assuming data is sorted and has a continuous index:
data = pd.read_csv('monthly_avg_temperatures.csv')

# Sort data by Year and Month if it's not sorted
data.sort_values(by=['Year', 'Month'], inplace=True)
data['Date'] = pd.to_datetime(data[['Year', 'Month']].assign(Day=1))
data.set_index('Date', inplace=True)

# Drop NA values generated from lagging
data.dropna(inplace=True)

# Split the data into train, validation, and test sets manually
train_size = int(len(data) * 0.6)
validation_size = int(len(data) * 0.2)
train = data.iloc[:train_size]
validation = data.iloc[train_size:train_size + validation_size]
test = data.iloc[train_size + validation_size:]

# Fit ARIMA model
model = ARIMA(train['AvgTemperature'], order=(24,0,24))  # Adjust order as necessary
fitted_model = model.fit()

# Predict on validation set
validation_start = validation.index[0]
validation_end = validation.index[-1]
validation_pred = fitted_model.predict(start=validation_start, end=validation_end, typ='levels')

# Predict on test set
test_start = test.index[0]
test_end = test.index[-1]
test_pred = fitted_model.predict(start=test_start, end=test_end, typ='levels')

# Calculate MSE
validation_mse = mean_squared_error(validation['AvgTemperature'], validation_pred)
test_mse = mean_squared_error(test['AvgTemperature'], test_pred)




# Fit the ARIMA model


# Predict on validation and test sets

# Print errors
#print(f'Validation MSE: {validation_mse}')
#print(f'Test MSE: {test_mse}')

# Calculate Mean Absolute Error (MAE)
validation_mae = np.mean(np.abs(validation['AvgTemperature'] - validation_pred))
test_mae = np.mean(np.abs(test['AvgTemperature'] - test_pred))

# Calculate Root Mean Squared Error (RMSE)
validation_rmse = np.sqrt(mean_squared_error(validation['AvgTemperature'], validation_pred))
test_rmse = np.sqrt(mean_squared_error(test['AvgTemperature'], test_pred))

# Calculate Mean Absolute Percentage Error (MAPE)
def calculate_mape(actual, predicted):
    return np.mean(np.abs((actual - predicted) / actual)) * 100

validation_mape = calculate_mape(validation['AvgTemperature'], validation_pred)
test_mape = calculate_mape(test['AvgTemperature'], test_pred)


# Print errors
print(f'Validation MSE: {validation_mse}')
print(f'Validation MAE: {validation_mae}')
print(f'Validation RMSE: {validation_rmse}')
print(f'Validation MAPE: {validation_mape}')

print(f'Test MSE: {test_mse}')
print(f'Test MAE: {test_mae}')
print(f'Test RMSE: {test_rmse}')
print(f'Test MAPE: {test_mape}')


# Visualization
# plt.figure(figsize=(12, 6))
# plt.plot(train.index, train, label='Training Data')
# plt.plot(validation.index, validation, label='Validation Data')
# plt.plot(validation.index, validation_pred, label='Validation Prediction', color='red')
# plt.plot(test.index, test, label='Test Data', color='green')
# plt.plot(test.index, test_pred, label='Test Prediction', color='orange')
# plt.legend()
# plt.show()
