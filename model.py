import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


data = pd.read_csv('data/Kendeda_Building_Auditorium_152_0c8b95e42928_31_Oct_2024.csv')

# inspect data
# print(data.head())
# print(data.info())

# Check for missing values
# print(data.isnull().sum())

# Fill missing values or drop rows/columns with missing data
data = data.fillna(method='ffill')  # Forward fill

data['dtm'] = pd.to_datetime(data['dtm'])
data.set_index('dtm', inplace=True)

features = ['temperature, °C', 'co2, ppm', 'humidity, %']
data = data[features]

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Define Sequence Length
sequence_length = 60  # Using past 60 minutes to predict the next value

# Create Sequences and Labels
def create_sequences(data, seq_length):
    x = []
    y = []
    for i in range(seq_length, len(data)):
        x.append(data[i-seq_length:i])
        y.append(data[i])
    return np.array(x), np.array(y)

X, y = create_sequences(scaled_data, sequence_length)

# Split Data into Training and Testing Sets
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build the LSTM Model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(units=50))
model.add(Dense(units=X_train.shape[2]))  # Output layer

model.compile(optimizer='adam', loss='mean_squared_error')

#Train the Model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the Model
# Plot Training and Validation Loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

# Make Predictions
predictions = model.predict(X_test)

# Inverse Transform Predictions
# Since we scaled the data, we need to inverse transform to get actual values
y_test_inverse = scaler.inverse_transform(y_test)
predictions_inverse = scaler.inverse_transform(predictions)

# Plot Actual vs Predicted
plt.figure(figsize=(12,6))
plt.plot(y_test_inverse[:, 0], color='blue', label='Actual Temperature')
plt.plot(predictions_inverse[:, 0], color='red', label='Predicted Temperature')
plt.title('Temperature Prediction')
plt.xlabel('Time')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.show()

# Predict Future IEQ Conditions
# Get the Last Sequence
last_sequence = scaled_data[-sequence_length:]
last_sequence = np.expand_dims(last_sequence, axis=0)

# Predict the Next Value
future_prediction = model.predict(last_sequence)
future_prediction_inverse = scaler.inverse_transform(future_prediction)
print("Future IEQ Conditions Prediction:")
print(f"Temperature: {future_prediction_inverse[0][0]:.2f} °C")
print(f"CO2: {future_prediction_inverse[0][1]:.2f} ppm")
print(f"Humidity: {future_prediction_inverse[0][2]:.2f} %")