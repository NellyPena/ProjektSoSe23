from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import Constants

df = pd.read_csv('DPW.DE(3).csv')
company = 'DPW.DE(3)'

dates = list(range(0,int(len(df))))
prices = df['Close']
#Fehlende Werte imputieren (NaN)
prices[np.isnan(prices)] = np.median(prices[~np.isnan(prices)])

#Plotten von Originaldaten
plt.plot(df['Close'], label='Close Price history')
plt.title('Linear Regression | Time vs. Price (Original Data)')
plt.legend()
plt.xlabel('Date Integer')
plt.show()

#In Numpy-Array konvertieren und reshape 
dates = np.asanyarray(dates)
prices = np.asanyarray(prices)
dates = np.reshape(dates,(len(dates),1))
prices = np.reshape(prices, (len(prices), 1))

#Pickle-Datei laden, um die zuvor gespeicherte Modellgenauigkeit zu erhalten
try:
  pickle_in = open("prediction.pickle", "rb")
  reg = pickle.load(pickle_in)
  xtrain, xtest, ytrain, ytest = train_test_split(dates, prices, test_size=0.2)
  best = reg.score(ytrain, ytest)
except:
  pass

#Das genaueste Modell erhalten
best = 0
for _ in range(100):
    xtrain, xtest, ytrain, ytest = train_test_split(dates, prices, test_size=0.2)
    reg = LinearRegression().fit(xtrain, ytrain)
    acc = reg.score(xtest, ytest)
    if acc > best:
        best = acc
        #Save model to pickle format
        with open('prediction.pickle','wb') as f:
            pickle.dump(reg, f)
        print(acc)

#Lineares Regressionsmodell laden
pickle_in = open("prediction.pickle", "rb")
reg = pickle.load(pickle_in)

#Durchschnittliche Genauigkeit des Modells 
mean = 0
for i in range(10):
  #Random Split Data
  msk = np.random.rand(len(df)) < 0.8
  xtest = dates[~msk]
  ytest = prices[~msk]
  mean += reg.score(xtest,ytest)

accuracy = mean/10
print("Average Accuracy:", accuracy)
accuracy = pd.DataFrame({accuracy})
accuracy.to_csv(f"Accuracy_LR_{company}.csv", sep=',',index=False,header=False)

predictions = reg.predict(xtest)

#Plot Predicted VS Actual Data
plt.plot(xtest, ytest, color='green',linewidth=1, label= 'Actual Price') #plotting the initial datapoints
plt.plot(xtest, predictions, color='blue', linewidth=3, label = 'Predicted Price') #plotting the line made by linear regression
plt.title('Linear Regression | Time vs. Price ')
plt.legend()
plt.xlabel('Date Integer')
plt.show()

print(predictions)
predictions = predictions.reshape(-1)
predictions = pd.DataFrame(data={"Prediction_LR" : predictions})
predictions.to_csv(f"Prediction_LR_{company}.csv", sep=',',index=False)

# Future-Days Vorhersage
future_predictions = np.array([])
last = xtest[-1]
for i in range(Constants.X_FUTURE):
  curr_prediction = reg.predict(np.array([last])).reshape(-1)
  print(curr_prediction)
  last = np.concatenate([last[1:], curr_prediction])
  future_predictions = np.concatenate([future_predictions, curr_prediction]) 
print(future_predictions)

#Plot Predicted VS Actual Data
plt.plot(future_predictions, color='blue', linewidth=3, label = 'Predicted Price') #Einzeichnen der Linie aus der linearen Regression
plt.title('Linear Regression | Time vs. Price ')
plt.legend()
plt.xlabel('Date Integer')
plt.show()