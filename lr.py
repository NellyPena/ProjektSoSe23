#LSTM-Based Code
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import numpy as np
from sklearn.linear_model import LinearRegression
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
# y = 1 * x_0 + 2 * x_1 + 3
y = np.dot(X, np.array([1, 2])) + 3
reg = LinearRegression().fit(X, y)
reg.score(X, y)
reg.coef_
reg.intercept_
predictions = reg.predict(np.array([[3, 5]]))

company = "Tsla"

#### PLOT
#plt.plot(future_predictions, color="tomato", label=f"{company} future X days prices")
plt.plot(predictions, color="steelblue", label=f"{company} Predicted Prices")
plt.title(f"{company} Share Price Vs Prediction") #plot not showing after adding this line
plt.legend()
plt.show()