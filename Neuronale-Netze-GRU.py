#LSTM-Based Code
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras 
from keras.models import Sequential, load_model
from keras.layers import GRU, Dense, Dropout
from sklearn.metrics import mean_squared_error

df = pd.read_csv('DAX.csv')
company = 'DAX'

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
    for i in range(50, df.shape[0]): #50 affects len(prediction) , len(prediction) = len(x_test) + len(window)
        x.append(df[i-50:i, 0])  #50
        y.append(df[i, 0])
    x = np.array(x)
    y = np.array(y)
    return x,y

x_train, y_train = create_dataset(dataset_train)
x_test, y_test = create_dataset(dataset_test)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

model = Sequential()
#layer specification
model.add(GRU(units=50, return_sequences = True, input_shape=(x_train.shape[1],1))) #units can be changed, return = true for recurrent neural networks (as LSTM feeds information back, not only forward)
#input shape is...
model.add(Dropout(0.2))
model.add(GRU(units=50, return_sequences = True))
model.add(Dropout(0.2))
model.add(GRU(units=50))
model.add(Dropout(0.1))
model.add(Dense(units=1)) #prediction of the next closing price

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=5, batch_size=32) #epochs25
#model.save('stock_prediction.h5')#model training operations, not required.

#model = load_model('stock_prediction.h5')

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

X_FUTURE = 50 #always using x[-1] from x_test for the prediction
future_predictions = np.array([])
last = x_test[-1]
for i in range(X_FUTURE):
  curr_prediction = model.predict(np.array([last]))
  print(curr_prediction)
  last = np.concatenate([last[1:], curr_prediction])
  future_predictions = np.concatenate([future_predictions, curr_prediction[0]])
future_predictions = scaler.inverse_transform([future_predictions])[0]
print(future_predictions)


#### Mean Squared Error
mse = mean_squared_error(y_test_scaled, predictions)
print("Mean Squared Error:", mse)

#### PLOT
plt.plot(future_predictions, color="tomato", label=f"{company} future X days prices")
plt.plot(y_test_scaled, color="black", label=f"{company} real prices")
plt.plot(predictions, color="steelblue", label=f"{company} predicted prices")
plt.title(f"{company} Share Price Vs Prediction") #plot not showing after adding this line
plt.legend()
plt.show()

#print(predicted_prices)
predictions = predictions.reshape(-1)
predictions = pd.DataFrame(data={"Prediction_GRU" : predictions})
#predicted_prices = np.asarray(predicted_prices)
print(predictions)
#predicted_prices.tofile('Prediction_neural.csv', sep = ',')
predictions.to_csv("Prediction_GRU.csv", sep=',',index=False)