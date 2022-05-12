# Development of Multivariate LSTM model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow

data = pd.read_csv('Electricity+Consumption.csv')
data.dropna(inplace=True)

sns.heatmap(data.corr())

train = data.iloc[:8712, 1:4].values
test = data.iloc[8712:, 1:4].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))

training_set_scaled = sc.fit_transform(train)
testing_set_scaled = sc.fit_transform(test)

testing_set_scaled = testing_set_scaled[:, 0:2]

x_train = []
y_train = []
ws = 24
for i in range(ws, len(training_set_scaled)):
    x_train.append(training_set_scaled[i-ws:i, 0:3])
    y_train.append(training_set_scaled[i, 2])

x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 3))

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout

model = Sequential()

model.add(LSTM(units=70, return_sequences=True, input_shape=(x_train.shape[1], 3)))
model.add(Dropout(0.2))
model.add(LSTM(units=70, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=70, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=70))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=80, batch_size=32)

plt.plot(range(len(model.history.history['loss'])), model.history.history['loss'])
plt.xlabel('Epoch NUmber')
plt.ylabel('loss')
plt.show()

model.save('LSTM - Multivariate')

from keras.models import load_model
model = load_model('LSTM - Multivariate')

prediction_test = []

batch_one = training_set_scaled[-24:]
batch_new = batch_one.reshape((1, 24, 3))

for i in range(48):
    first_prediction = model.predict(batch_new)[0]
    prediction_test.append(first_prediction)
    new_var = testing_set_scaled[i, :]
    new_var = new_var.reshape(1, 2)
    new_test = np.insert(new_var, 2, [first_prediction], axis=1)
    new_test = new_test.reshape(1, 1, 3)
    batch_new = np.append(batch_new[:, 1:, :], new_test, axis=1)

prediction_test = np.array(prediction_test)

si = MinMaxScaler(feature_range=(0, 1))
y_scale = train[:, 2:3]
si.fit_transform(y_scale)

predictions = si.inverse_transform(prediction_test)

real_values = test[:, 2]

plt.plot(real_values, color='red', label='Actual Electrical Consumption')
plt.plot(predictions, color='blue', label='predicted values')
plt.title('Electrical Consumption Prediction')
plt.xlabel('Time(hr)')
plt.ylabel('Electrical Demand (MW)')
plt.legend()
plt.show()

import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(test, predictions))

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs(y_true-y_pred)/y_true)*100

MAPE = mean_absolute_percentage_error((real_values, predictions))





