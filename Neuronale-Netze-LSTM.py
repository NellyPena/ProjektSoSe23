import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras 
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error

import Constants

df = pd.read_csv('MDO.DE(5Y).csv')
company = 'MDO.DE(5Y)'

#Parameter
window_size = 20
noepochs = 100 
nobatchsize = 32 
nounits = 100 
dropout= 0 
training_size = 0.8

df = df['Close'].values #Open
df = df.reshape(-1, 1)

dataset_train = np.array(df[:int(df.shape[0]*training_size)])
dataset_test = np.array(df[int(df.shape[0]*training_size):]) 

# Datavorbereitung
scaler = MinMaxScaler(feature_range=(0,1))
dataset_train = scaler.fit_transform(dataset_train)
dataset_test = scaler.transform(dataset_test)

#Dataset-windowing
def create_dataset(df):
    x = []
    y = []
    for i in range(window_size, df.shape[0]): 
        x.append(df[i-window_size:i, 0])  
        y.append(df[i, 0])
    x = np.array(x)
    y = np.array(y)
    return x,y

x_train, y_train = create_dataset(dataset_train)
x_test, y_test = create_dataset(dataset_test)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

model = Sequential()

#Layer specification
model.add(LSTM(units=nounits, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(dropout))
model.add(LSTM(units=nounits,return_sequences=True))
model.add(Dropout(dropout))
model.add(LSTM(units=nounits,return_sequences=True))
model.add(Dropout(dropout))
model.add(LSTM(units=nounits))
model.add(Dropout(dropout))
model.add(Dense(units=1))

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=noepochs, batch_size=nobatchsize) 
model.save('stock_prediction_LSTM.h5')#model training operations

model = load_model('stock_prediction_LSTM.h5')

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

print(x_test.shape)

#Future-Days Vorhersage
future_predictions = np.array([])
last = x_test[-1]
for i in range(Constants.X_FUTURE):
  curr_prediction = model.predict(np.array([last]))
  print(curr_prediction)
  last = np.concatenate([last[1:], curr_prediction])
  future_predictions = np.concatenate([future_predictions, curr_prediction[0]])
future_predictions = scaler.inverse_transform([future_predictions])[0]
print(future_predictions)

#Linienverbindung und Future-Days Vorhersage CSV
dicts = []
last_day = len(y_test)-1 
for i in range(Constants.X_FUTURE):
  last_day = last_day + 1
  dicts.append({'Predictions':future_predictions[i], "Date": last_day})

new_data = pd.DataFrame(dicts).set_index("Date")

future_predictions = future_predictions.reshape(-1)
future_predictions = pd.DataFrame(data={"Prediction_LSTM" : future_predictions})
print(future_predictions)
future_predictions.to_csv(f"FuturePrediction_LSTM_{company}.csv", sep=',',index=False)
print("--------------------------------")
print(len(future_predictions), Constants.X_FUTURE)
print("--------------------------------")

# Mean Squared Error + .csv
mse = mean_squared_error(y_test_scaled, predictions)
print("Mean Squared Error:", mse)
mse = pd.DataFrame({mse})
mse.to_csv(f"MSE_LSTM_{company}.csv", sep=',',index=False,header=False)

# PLOT
plt.plot(y_test_scaled, color="black", label=f"{company} real prices")
plt.plot(predictions, color="steelblue", label=f"{company} predicted prices")
plt.plot(new_data, color="tomato", label=f"{company} Future {Constants.X_FUTURE} Days")
plt.title(f"{company} Share Price Vs Prediction") #plot not showing after adding this line
plt.legend()
plt.show()

#CSV der vorhergesagten Daten
predictions = predictions.reshape(-1)
predictions = pd.DataFrame(data={"Prediction_LSTM" : predictions})
print(predictions)
predictions.to_csv(f"Prediction_LSTM_{company}.csv", sep=',',index=False)
