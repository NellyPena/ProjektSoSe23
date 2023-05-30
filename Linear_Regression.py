import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

dataset=pd.read_csv('TSLA.csv')
X=dataset[['Open','High','Close']]
y=dataset['Close']

dataset.head()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X,y, random_state=0)

print(X_train.shape)
print(X_test.shape)

print(X_test)
print("xtrain")
print(X_train)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, accuracy_score
regressor= LinearRegression()

regressor.fit(X_train,y_train)
predicted=regressor.predict(X_test)
dframe=pd.DataFrame(y_test,predicted)

print(y_test)
print(y_train)
print(predicted)
dfr=pd.DataFrame({'Actual=':y_test,'Predicted':predicted})

print(dfr)

from sklearn.metrics import confusion_matrix, accuracy_score
import math

graph=dfr.head(20)
graph.plot(kind='bar')