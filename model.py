# -*- coding: utf-8 -*-
"""
Spyder Editor - Matthew Xie

This is the Stock Market Analysis. 

We conduct both decriptive analysis and inferential analysis
by using Monte Carlo method and Machine Learning regression models.

There should've been done with Time Series Analysis to verify
if the data is stationary. Due to time reason, 
we didn't implement.

As a result, the accuracy of prediction is not high because 
the adfuller test is not conducted.
"""
#%%
# Try to download and save stock data 
import yfinance as yf  
data_df = yf.download('XL')
data_df.to_csv('XL.csv')
#%%

import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

# Pick up stock and period set to analyse
stock_list = ['OPTT','BB','XL','HYLN']
start = datetime.datetime(2010, 12, 23)
end = datetime.date.today()

for stock in stock_list:
    globals()[stock] = yf.download(stock, start, end)
#%%
# Plot some basic descriptive analysis
XL['Adj Close'].plot()
XL['Volume'].plot(label = 'Volume',legend = True,figsize=(10,4))
# Moving average of 10, 20, 50 days
ma_day = [10,20,50]
for ma in ma_day:
    new_column = 'MA for {} days'.format(str(ma))
    XL[new_column] = XL['Adj Close'].rolling(window=ma).mean()
XL[['Adj Close','MA for 10 days','MA for 20 days','MA for 50 days']]\
.plot(figsize=(10,6))
#%%
# Dayly Return Analysis
XL['Daily Return'] = XL['Adj Close'].pct_change()
XL['Daily Return'].plot(linestyle = '-', marker='o',figsize = (10,5),title = 'Daily Return')

# KDE plot
sns.distplot(XL['Daily Return'].dropna(),color = 'purple',\
             norm_hist=False,kde=False,bins=10)
    
# Cumulative Return 
((1 + XL['Daily Return']).cumprod() - 1).plot(title='XL Cumulative Returns')

# Analysis four stocks based on closing price
from pandas_datareader import data
closing_df = data.DataReader(['OPTT','BB','XL','HYLN'],'yahoo',start,end)['Adj Close']
stock_daily_return = closing_df.pct_change()
#%%
# Compare daily returns of different stocks
sns.jointplot('OPTT','XL',stock_daily_return.dropna(), kind='scatter',color='green')
# Compare all four stocks
sns.pairplot(stock_daily_return)
stock_compare = sns.PairGrid(stock_daily_return.dropna())
stock_compare.map_lower(sns.kdeplot,cmap='Blues_d')
stock_compare.map_upper(plt.scatter,color='purple')
stock_compare.map_diag(plt.hist)
#%%
# Compare closing prices of the four stocks
closing_compare = sns.PairGrid(closing_df.dropna())
closing_compare.map_diag(plt.hist,bins=30)
closing_compare.map_upper(plt.scatter,color='purple')
closing_compare.map_lower(sns.kdeplot,cmap='cool_d')
#%%
# Correlation plot
corr = closing_df.dropna().corr()
sns.heatmap(corr,annot=True)
corr2 = stock_daily_return.dropna().corr()
sns.heatmap(corr2,annot=True)
#%%
# Risk Analysis
from sklearn import preprocessing
# Normalise Data
x=stock_daily_return.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled,columns=('OPTT','BB','XL','HYLN'))
rets = df.dropna()
plt.scatter(rets.mean(),rets.std(),alpha=0.5,\
        
            )

plt.xlabel('Expected returns')
plt.ylabel('Risk')
plt.ylim([0.1,0.25])
plt.xlim([0.18,0.65])
for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
    plt.annotate(
        label, 
        xy = (x, y), xytext = (50, 50),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=-0.3',\
        color='black'))
# With 99% confidence, the worst laily loss won't exceed ()
stock_daily_return['XL'].quantile(0.01)
#%%
# Value risk using the Monte Carlo method

days = 365
d = 1 /365
mean = stock_daily_return.mean()['XL']
std = stock_daily_return.std()['XL']

def Monte_Carlo(start_price,days,mean,std):
    price = np.zeros(days)
    price[0] = start_price
    shock = np.zeros(days)
    drift = np.zeros(days)
    
    for x in range(1,days):
        shock[x] = np.random.normal(loc=mean*d,scale=std*np.sqrt(d))
        drift[x] = mean*d
        price[x] = price[x-1] + (price[x-1] * (drift[x] + shock[x]))
    return price

start_price = round(OPTT.head().Open[0],2)

for run in range(100):
    plt.plot(Monte_Carlo(start_price, days, mean, std))
plt.xlabel('Days')
plt.ylabel('Price')
plt.title('Monte Carlo Analysis for XL Fleet')

# Set a large numebr of runs
runs = 10000

# Create an empty matrix to hold the end price data
simulations = np.zeros(runs)

# Set the print options of numpy to only display 0-5 points from an array to suppress output
np.set_printoptions(threshold=5)

for run in range(runs):    
    # Set the simulation data point as the last stock price for that run
    simulations[run] = Monte_Carlo(start_price,days,mean,std)[days-1];

# Now we'll define q as the 1% empirical qunatile, this basically means that 99% of the values should fall between here
q = np.percentile(simulations, 1)
    
# Now let's plot the distribution of the end prices
plt.hist(simulations,bins=200)

# Using plt.figtext to fill in some additional information onto the plot

# Starting Price
plt.figtext(0.6, 0.8, s="Start price: $%.2f" %start_price)
# Mean ending price
plt.figtext(0.6, 0.7, "Mean final price: $%.2f" % simulations.mean())

# Variance of the price (within 99% confidence interval)
plt.figtext(0.6, 0.6, "VaR(0.99): $%.2f" % (start_price - q,))

# Display 1% quantile
plt.figtext(0.15, 0.6, "q(0.99): $%.2f" % q)

# Plot a line at the 1% quantile result
plt.axvline(x=q, linewidth=4, color='r')

# Title
plt.title(u"Final price distribution for XL Stock after %s days" % days, weight='bold')
#%% 
# Inferential Analysis of predicting stock prices
predict_period = 5
XL = yf.download('XL', start,end)
XL['Prediction'] = XL['Adj Close'].shift(-predict_period)
x = np.array(XL.drop(['Prediction'],axis=1))[:-predict_period]
y = np.array(XL['Prediction'])[:-predict_period]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)
#%%
# Linear Regression 
from sklearn.linear_model import LinearRegression, Lasso, Ridge
linear_model = LinearRegression()
linear_model.fit(x_train, y_train)
# Score the model (Test data)
linear_model_score = linear_model.score(x_test, y_test)
print('Linear Model Score:', linear_model_score)
# Define the Real & Prediction Values
X_predict = np.array(XL.drop(['Prediction'], 1))[-predict_period:]

linear_model_predict_prediction = linear_model.predict(X_predict)
linear_model_real_prediction = linear_model.predict(np.array(XL.drop(['Prediction'], 1)))

# Defining some Parameters
from datetime import timedelta, datetime
predicted_dates = []
recent_date = BB.index.max()
alpha = 0.5

for i in range(predict_period):
    recent_date += timedelta(days=1)
    predicted_dates.append(recent_date)

# Plotting the Actual and Prediction Prices
plt.figure(figsize=(20, 12))
plt.plot(XL.index, linear_model_real_prediction, label='Linear Prediction', color='blue', alpha=alpha)
plt.plot(predicted_dates, linear_model_predict_prediction, label='Forecast', color='green', alpha=alpha)
plt.plot(XL.index, XL['Close'], label='Actual', color='red')
plt.legend()

#%%
# Ridge Regression
# Defining the Ridge Regression Model
ridge_model = Ridge()
ridge_model.fit(x_train, y_train)   

ridge_model_score = ridge_model.score(x_test, y_test)
print('Ridge Model score:', ridge_model_score)

# Define the Real & Prediction Values
ridge_model_predict_prediction = ridge_model.predict(X_predict)
ridge_model_real_prediction = ridge_model.predict(np.array(XL.drop(['Prediction'], 1)))

# Plotting the Actual and Prediction Prices
plt.figure(figsize=(20, 10))
plt.plot(XL.index, ridge_model_real_prediction, label='Ridge Prediction', color='blue', alpha=alpha)
plt.plot(predicted_dates, ridge_model_predict_prediction, label='Forecast', color='green', alpha=alpha)
plt.plot(XL.index, XL['Close'], label='Actual', color='red')
plt.legend()

#%%
# Lasso Regression
# Defining the Lasso Regression Model
lasso_model = Lasso()
lasso_model.fit(x_train, y_train)   

lasso_model_score = lasso_model.score(x_test, y_test)
print('Lasso Model score:', lasso_model_score)

lasso_model_predict_prediction = lasso_model.predict(X_predict)
lasso_model_real_prediction = lasso_model.predict(np.array(XL.drop(['Prediction'], 1)))

# Plotting the Actual and Prediction Prices
plt.figure(figsize=(20, 10))
plt.plot(XL.index, lasso_model_real_prediction, label='Lasso Prediction', c='blue', alpha=alpha)
plt.plot(predicted_dates, lasso_model_predict_prediction, label='Forecast', color='green', alpha=alpha)
plt.plot(XL.index, XL['Close'], label='Actual', color='red')
plt.legend()

#%%
# Best Performance of the Regressor Models
best_score = max(linear_model_score, ridge_model_score, lasso_model_score)
index = np.argmax([linear_model_score, ridge_model_score, lasso_model_score])
best_regressor = {0:'Linear Regression Model',
                  1:'Ridge Model',
                  2:'Lasso Model'}
print("The Best Performer is {0} with the score of {1}%.".format(best_regressor[index], best_score*100))
