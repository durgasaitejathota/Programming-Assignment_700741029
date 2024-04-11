import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
import matplotlib.pyplot as plt

# Load the dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
df = pd.read_csv(url, usecols=[1], engine='python')
dataset = df.values.astype('float32')

# Normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# Split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# Convert an array of values into a dataset matrix
def create_dataset(dataset, time_steps=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_steps-1):
        a = dataset[i:(i+time_steps), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_steps, 0])
    return np.array(dataX), np.array(dataY)

# Reshape into X=t and Y=t+1
time_steps = 1
X_train, y_train = create_dataset(train, time_steps)
X_test, y_test = create_dataset(test, time_steps)

# Reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# Define LSTM model
def build_lstm_model(input_shape, num_units=[50, 50], dropout_rate=0.2):
    model = Sequential()
    model.add(LSTM(units=num_units[0], input_shape=input_shape, return_sequences=True))
    model.add(Dropout(dropout_rate))
    if len(num_units) > 1:
        for units in num_units[1:]:
            model.add(LSTM(units=units, return_sequences=True))
            model.add(Dropout(dropout_rate))
    model.add(LSTM(units=num_units[-1]))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    return model

# Build the model
model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]), num_units=[50, 50], dropout_rate=0.2)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=1, validation_data=(X_test, y_test), verbose=2)

# Plot training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transformation to get original values
train_predict = scaler.inverse_transform(train_predict)
y_train = scaler.inverse_transform([y_train])
test_predict = scaler.inverse_transform(test_predict)
y_test = scaler.inverse_transform([y_test])

# Calculate RMSE
train_score = np.sqrt(mean_squared_error(y_train[0], train_predict[:,0]))
print('Train RMSE: %.2f' % (train_score))
test_score = np.sqrt(mean_squared_error(y_test[0], test_predict[:,0]))
print('Test RMSE: %.2f' % (test_score))

# Plot predictions
train_predict_plot = np.empty_like(dataset)
train_predict_plot[:, :] = np.nan
train_predict_plot[time_steps:len(train_predict)+time_steps, :] = train_predict

test_predict_plot = np.empty_like(dataset)
test_predict_plot[:, :] = np.nan
test_predict_plot[len(train_predict)+(time_steps*2)+1:len(dataset)-1, :] = test_predict

plt.plot(scaler.inverse_transform(dataset), label='Original Data')
plt.plot(train_predict_plot, label='Train Predictions')
plt.plot(test_predict_plot, label='Test Predictions')
plt.xlabel('Months')
plt.ylabel('Passenger Count')
plt.title('Airline Passengers Prediction')
plt.legend()
plt.show()
