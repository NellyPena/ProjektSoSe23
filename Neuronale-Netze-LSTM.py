import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras 
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error

df = pd.read_csv('ALV.csv')
company = 'ALV'
window_size = 50

df = df['Close'].values #Open
df = df.reshape(-1, 1)

dataset_train = np.array(df[:int(df.shape[0]*0.7)])
dataset_test = np.array(df[int(df.shape[0]*0.7):])

# preparar los datos
scaler = MinMaxScaler(feature_range=(0,1))
dataset_train = scaler.fit_transform(dataset_train)
dataset_test = scaler.transform(dataset_test)

def create_dataset(df):
    x = []
    y = []
    for i in range(window_size, df.shape[0]): #50
        x.append(df[i-window_size:i, 0])  #50
        y.append(df[i, 0])
    x = np.array(x)
    y = np.array(y)
    return x,y

x_train, y_train = create_dataset(dataset_train)
x_test, y_test = create_dataset(dataset_test)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

model = Sequential()
model.add(LSTM(units=96, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=96,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=96,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=96))
model.add(Dropout(0.2))
model.add(Dense(units=1))

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=5, batch_size=32) #epochs25
model.save('stock_prediction_LSTM.h5')#model training operations, not required.

model = load_model('stock_prediction_LSTM.h5')

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

print(x_test.shape)

X_FUTURE = 5
future_predictions = np.array([])
last = x_test[-1]
for i in range(X_FUTURE):
  curr_prediction = model.predict(np.array([last]))
  print(curr_prediction)
  last = np.concatenate([last[1:], curr_prediction])
  future_predictions = np.concatenate([future_predictions, curr_prediction[0]])
future_predictions = scaler.inverse_transform([future_predictions])[0]
print(future_predictions)

###join lines 
dicts = []
last_day = len(y_test)-1 #-1 -2 reduces gap but means prediction on same day from both lines
for i in range(X_FUTURE):
  last_day = last_day + 1
  dicts.append({'Predictions':future_predictions[i], "Date": last_day})

new_data = pd.DataFrame(dicts).set_index("Date")

#### Mean Squared Error
mse = mean_squared_error(y_test_scaled, predictions)
print("Mean Squared Error:", mse)

#### PLOT
#plt.plot(future_predictions, color="tomato", label=f"{company} future X days prices")
plt.plot(y_test_scaled, color="black", label=f"{company} real prices")
plt.plot(predictions, color="steelblue", label=f"{company} predicted prices")
plt.plot(new_data, color="tomato", label=f"{company} Future {X_FUTURE} Days")
plt.title(f"{company} Share Price Vs Prediction") #plot not showing after adding this line
plt.legend()
plt.show()

#print(predicted_prices)
predictions = predictions.reshape(-1)
predictions = pd.DataFrame(data={"Prediction_LSTM" : predictions})
#predicted_prices = np.asarray(predicted_prices)
print(predictions)
#predicted_prices.tofile('Prediction_neural.csv', sep = ',')
predictions.to_csv("Prediction_LSTM.csv", sep=',',index=False)