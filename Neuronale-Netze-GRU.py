import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras 
from keras.models import Sequential, load_model
from keras.layers import GRU, Dense, Dropout
from sklearn.metrics import mean_squared_error

import Constants 

df = pd.read_csv('^MDAXI_10y.csv')
company = 'tuning_^MDAXI.DE(10Y)'

#Parameter
window_size = 60 
noepochs = 100 
nobatchsize = 64 
nounits = 100 
dropout= 0 
training_size = 0.8

df = df['Close'].values
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
    for i in range(window_size, df.shape[0]): #len(prediction) = len(x_test) + len(window)
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
model.add(GRU(units=nounits, return_sequences = True, input_shape=(x_train.shape[1],1))) 
model.add(Dropout(dropout))
model.add(GRU(units=nounits, return_sequences = True))
model.add(Dropout(dropout))
model.add(GRU(units=nounits))
model.add(Dropout(dropout))
model.add(Dense(units=1)) #prediction of the next closing price

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=noepochs, batch_size=nobatchsize) 
model.save('stock_prediction_GRU.h5') #model training operations

model = load_model('stock_prediction_GRU.h5')

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

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

#### Mean Squared Error + .csv
mse = mean_squared_error(y_test_scaled, predictions)
print("Mean Squared Error:", mse)
mse = pd.DataFrame({mse})
mse.to_csv(f"MSE_GRU_{company}.csv", sep=',',index=False,header=False)

#### PLOT
plt.plot(y_test_scaled, color="black", label=f"{company} Real Prices")
plt.plot(predictions, color="steelblue", label=f"{company} Predicted Prices")
plt.plot(new_data, color="tomato", label=f"{company} Future {Constants.X_FUTURE} Days")
plt.title(f"{company} Share Price Vs Prediction")
plt.legend()
plt.show()

#CSV der vorhergesagten Daten
predictions = predictions.reshape(-1)
predictions = pd.DataFrame(data={"Prediction_GRU" : predictions})
print(predictions)
predictions.to_csv(f"Prediction_GRU_{company}.csv", sep=',',index=False)

