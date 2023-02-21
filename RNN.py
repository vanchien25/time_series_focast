import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense,GRU,RNN
import pandas as pd
from matplotlib import pyplot as plt
from keras.layers import SimpleRNNCell, RNN
# Load the data and split into training and test sets
# Load the dataset
df = pd.read_csv('/content/BVData_csv.csv')
data = df.iloc[:,4:26]
# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Split the data into training and testing sets
train_data = scaled_data[:int(len(scaled_data)*0.8), :]
test_data = scaled_data[int(len(scaled_data)*0.8):, :]

# Function to create training data for multi-step prediction
def create_multistep_training_data(data, lookback, steps_ahead):
    X, y = [], []
    for i in range(len(data) - lookback - steps_ahead + 1):
        X.append(data[i:i + lookback, :])
        y.append(data[i + lookback:i + lookback + steps_ahead,-1].flatten())
    return np.array(X), np.array(y)
# Define the input sequence length (lookback) and number of time steps to predict ahead (steps_ahead)
lookback = 12
steps_ahead = 9

# Create the training data
X_train, y_train = create_multistep_training_data(train_data, lookback, steps_ahead)
# Define the LSTM model
n_features = data.shape[1]
model = Sequential()
model.add(RNN(SimpleRNNCell(100), input_shape=(lookback, n_features)))
model.add(Dense(steps_ahead))
model.compile(optimizer='adam', loss='mean_squared_error')

# Fit the model to the training data
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1)
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()

# Use the model to make predictions on the test data
X_test, y_test = create_multistep_training_data(test_data, lookback, steps_ahead)
y_pred = model.predict(X_test)
y_test_inv = y_test[:,:1]*(2100-10)+10
y_pred_inv = y_pred[:,:1]*(2100-10)+10

from sklearn.metrics import mean_squared_error

# Calculate RMSE and MSE for multi-step prediction
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
mse = mean_squared_error(y_test_inv, y_pred_inv)
nse = 1 - np.sum((y_pred_inv - y_test_inv) ** 2) / np.sum((y_test_inv - np.mean(y_test_inv)) ** 2)
print("RMSE for multi-step prediction:", rmse)
print("MSE for multi-step prediction:", mse)
print("NSE = ", nse)
