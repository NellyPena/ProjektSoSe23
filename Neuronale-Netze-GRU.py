import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras 
from keras.models import Sequential
from keras.layers import Dense,Dropout,GRU

#Cargar los datos
company = 'TSLA' #used in the graph plot
hist = pd.read_csv('TSLA.csv') #2012 - 2019
df = pd.read_csv('TSLA.csv')
#hist =np.array(df[:int(df.shape[0]*0.7)])

print(hist)

#preparar los datos 
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(hist['Close'].values.reshape(-1,1))

prediction_days = 60

x_train = [] #prepared training data
y_train = [] #prepared training data

for x in range(prediction_days, len(scaled_data)): #from prediction days value until the end of the train test array
    x_train.append(scaled_data[x-prediction_days:x, 0]) #we take the first 60 values
    y_train.append(scaled_data[x,0]) #the 61st value

x_train, y_train = np.array(x_train),np.array(y_train) # converted into numpy arrays
x_train = np.reshape(x_train,(x_train.shape[0], x_train.shape[1],1)) #reshape so it can work with the neural network

print(x_train.shape)

#construir el modelo
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

#many layers lead to overfitting

model.compile(optimizer='adam', loss='mean_squared_error') #loss function : mean squared error

model.fit(x_train,y_train, epochs=25, batch_size=32) #batch size: the model is going to see 32 units at once all the time, 

#######################################test model accuracy on existing data
#cargar los datos de test
hist_test = pd.read_csv('TSLA_testdata.csv') #2020 to date
#hist_test = np.array(df[int(df.shape[0]*0.7):]) #new data, not used before

actual_prices  = hist_test['Close'].values

#we need to get and scale the prices
total_dataset = pd.concat((hist['Close'],hist_test['Close']),axis=0)  #data set that combines the closing data (not-scaled) with closed values test data
model_inputs = total_dataset[len(total_dataset)-len(hist_test)-prediction_days:].values #we want to start as soon as possible
model_inputs = scaler.transform(model_inputs.reshape(-1,1))  #to scale down

#### prediction based on the test data
x_test = []

for x in range(prediction_days,len(model_inputs)): # len(model_inputs)+1 to get the newest one?
    x_test.append(model_inputs[x-prediction_days:x,0])

x_test = np.array(x_test) #transform to array
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1)) #same format as x_train

#### actual prediction
predicted_prices = model.predict(x_test) #remember: the prediction prices are scaled
predicted_prices = scaler.inverse_transform(predicted_prices) #un-scale the predicted prices

##plot
plt.plot(actual_prices, color="black", label=f"{company} real prices")
plt.plot(predicted_prices, color="blue", label=f"{company} predicted prices")
plt.title(f"{company} Share Price Vs Prediction") #plot not showing after adding this line
plt.legend()
plt.show()

#PREDICT NEXT DAY
real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs + 1),0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data,(real_data.shape[0],real_data.shape[1],1))

prediction =model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f"Prediction (for the next day): {prediction}")

##rentability, only when going LONG
rentability = 1
for i in range(1,len(actual_prices)):
    if predicted_prices[i] > actual_prices[i-1]:
        rentability*= actual_prices[i]/actual_prices[i-1]

print((rentability-1)*100,"%")

#print(predicted_prices)
predicted_prices = predicted_prices.reshape(-1)
predicted_prices = pd.DataFrame(data={"Prediction_GRU" : predicted_prices})
#predicted_prices = np.asarray(predicted_prices)
print(predicted_prices)
predicted_prices.to_csv("Prediction_GRU.csv", sep=',',index=False)