import yfinance as yf
from sklearn.linear_model import LinearRegression

# Fetch historical data from Yahoo Finance
data = yf.download('AAPL', start='2010-01-01', end='2023-01-01')

# Preprocess the data
data['Date'] = data.index
data['Date'] = data['Date'].astype(str)
data['Year'] = data['Date'].str[:4]
data['Month'] = data['Date'].str[5:7]
data['Day'] = data['Date'].str[8:10]

# Split the data into training and testing sets
train_data = data[data['Year'] < '2022']
test_data = data[data['Year'] >= '2022']

# Prepare the input features and target variable
train_X = train_data[['Year', 'Month', 'Day']].values.astype(int)
train_y = train_data['Close'].values

# Create and train the linear regression model
model = LinearRegression()
model.fit(train_X, train_y)

# Predict using the test data
test_X = test_data[['Year', 'Month', 'Day']].values.astype(int)
predictions = model.predict(test_X)

# Print the predicted values
for date, prediction in zip(test_data.index, predictions):
    print(f"Date: {date}, Predicted Close Price: {prediction:.2f}")
