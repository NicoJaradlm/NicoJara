import pandas as pd
import numpy as np  
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Load datasets
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
store_data = pd.read_csv('store.csv')

# Merge the train/test data with store data
train_data = pd.merge(train_data, store_data, on='Store')
test_data = pd.merge(test_data, store_data, on='Store')

# Convert the Date column to datetime format and sort by date
train_data['Date'] = pd.to_datetime(train_data['Date'])
train_data.sort_values('Date', inplace=True)

# Select a single store (for simplicity)
store_sales = train_data[train_data['Store'] == 1].copy()

# Create time-based features *after selecting store-specific data*
store_sales['DayOfWeek'] = store_sales['Date'].dt.dayofweek  # Monday = 0, Sunday = 6
store_sales['Month'] = store_sales['Date'].dt.month  # Month as a number
store_sales['WeekOfYear'] = store_sales['Date'].dt.isocalendar().week  # Week number of the year

# Create lagged features and rolling averages
store_sales['lag_1'] = store_sales['Sales'].shift(1)
store_sales['rolling_7'] = store_sales['Sales'].rolling(window=7).mean()
store_sales['rolling_30'] = store_sales['Sales'].rolling(window=30).mean()

# Fill missing values created by lagging and rolling
store_sales.fillna(0, inplace=True)

# Label Encoding or One-Hot Encoding for Categorical Columns
# For simplicity, let's label encode 'StateHoliday'
store_sales['StateHoliday'] = store_sales['StateHoliday'].astype(str)

label_encoder = LabelEncoder()
store_sales['StateHoliday'] = label_encoder.fit_transform(store_sales['StateHoliday'])

# Normalize the sales data (use scaler only on Sales)
scaler = MinMaxScaler(feature_range=(0, 1))
store_sales['Sales'] = scaler.fit_transform(store_sales[['Sales']])

# Use the features in the model input, including Sales and other relevant columns
X = store_sales[['Sales', 'Promo', 'StateHoliday', 'SchoolHoliday', 'DayOfWeek', 'Month', 'WeekOfYear']].values

# Function to create sequences for the LSTM model
def create_sequences(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps - 1):
        X.append(data[i:(i + time_steps), :])  
        y.append(data[i + time_steps, 0])  # Target (Sales) is still the first column
    return np.array(X), np.array(y)

# create sequences from the updated X
time_steps = 90 # Adjust as needed for how back your want model to look back at data
X_sequences, y_sequences = create_sequences(X, time_steps)

# Reshape X_sequences for LSTM input [samples, time steps, features]
X_sequences = X_sequences.reshape((X_sequences.shape[0], X_sequences.shape[1], X_sequences.shape[2]))


#train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_sequences, y_sequences, test_size=0.2, random_state=42)



#build an train LSTM Model 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Ensure all data is float32
X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.float32)
X_test = X_test.astype(np.float32)
y_test = y_test.astype(np.float32)

# Ensure no NaN values
X_train = np.nan_to_num(X_train)
y_train = np.nan_to_num(y_train)

# Build the LSTM model
from tensorflow.keras.layers import Bidirectional
#from tensorflow.keras.callbacks import ReduceLROnPlateau

# Get the number of features from the X_sequences shape
n_features = X_sequences.shape[2]  # This gives you the number of features (7)

# Build the LSTM model
model = Sequential()

# Adjust input_shape to use n_features instead of 1
model.add(LSTM(units=200, return_sequences=True, input_shape=(time_steps, n_features)))
model.add(Dropout(0.4))

model.add(LSTM(units=150, return_sequences=False))
model.add(Dropout(0.4))

model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Make predictions
predictions = model.predict(X_test)

# Inverse the normalization to get actual sales values
predictions = scaler.inverse_transform(predictions)
y_test_actual = scaler.inverse_transform([y_test])

# Plot actual vs predicted sales
import matplotlib.pyplot as plt

plt.plot(y_test_actual[0], label='Actual Sales')
plt.plot(predictions, label='Predicted Sales')
plt.title('Sales Forecasting with LSTM')
plt.xlabel('Days')
plt.ylabel('Sales')
plt.legend()
plt.show()


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Calculate MAE
mae = mean_absolute_error(y_test_actual[0], predictions)
print(f'Mean Absolute Error (MAE): {mae}')

# Calculate MSE
mse = mean_squared_error(y_test_actual[0], predictions)
print(f'Mean Squared Error (MSE): {mse}')

# Calculate RMSE
rmse = np.sqrt(mse)
print(f'Root Mean Squared Error (RMSE): {rmse}')

# Calculate R-squared
r2 = r2_score(y_test_actual[0], predictions)
print(f'R-squared (R²): {r2}')



# Prepare the data for visualization
results_df = pd.DataFrame({
    'Date': store_sales['Date'].values[-len(predictions):],  # Ensure this matches the length of predictions
    'Actual Sales': y_test_actual[0],
    'Predicted Sales': predictions.flatten()  # Flatten predictions to make it a 1D array
})

# Export the data to a CSV file
results_df.to_csv('sales_forecast_results.csv', index=False)
