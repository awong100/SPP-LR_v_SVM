import quandl
import pandas_datareader as web
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

df = web.DataReader('TSLA', data_source='yahoo', start='2012-01-01', end='2021-01-14')
#print first few rows
# print(df.head(5))
dataframe = df

#begin prediction
#number of days in advance i'd like to predict
numDaysOut = 15
#create another column in df that will hold the predicted value
df['Prediction'] = df[['Adj Close']].shift(-numDaysOut)
#print(df.tail())

### Independent dataset
# convert dataframe to numpy array
#remove the prediction column for this
x = np.array(df.drop(['Prediction'], 1))
#remove the last numDaysOut rows
x = x[:-numDaysOut]
#print(x)

### Create Dependent dataset
#Convert the dataframe to numpy array
#repeats like x
y = np.array(df['Prediction'])
y = y[:-numDaysOut]
#print(y)

# 80/20 split training/test
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.20)
# print(ytest)

#create and train our model
#SVM(Regression model(getting some numeric outpot))
svr_rbf = SVR(kernel='rbf', C=1000, gamma="scale")
svr_rbf.fit(xtrain, ytrain)
#print(svr_rbf)

#Test our model
#Score: returns the coeff of determination R^2f the prediction
#Best possible: 1
svr_confidence = svr_rbf.score(xtest, ytest)
print("SVR Confidence: ", svr_confidence)

#Create and train a LinearRegression model
regressor = LinearRegression()
regressor.fit(xtrain, ytrain)

#test the LinearRegression Model
lr_confidence = regressor.score(xtest, ytest)
print("LinearRegression confidence: ", lr_confidence)

#Set x_forecast equal to last 30 days from original
x_forecast = np.array(df.drop(['Prediction'], 1))[-numDaysOut:]
#print("x-forecast", x_forecast)

#Print the predictions fro the next numDaysOut days SVR
svr_prediction = svr_rbf.predict(x_forecast)
# print("SVR Prediction: ", svr_prediction)

#Print the predictions fro the next numDaysOut days LP
lr_prediction = regressor.predict(x_forecast)
# print("LinearRegression Prediction: ", lr_prediction)

#print(df.tail())

#Plot the data
plt.figure(figsize=(16,8))
plt.title("Predicted Close Price - LR model")
plt.plot(lr_prediction)
plt.xlabel('n days from today', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.show( )



