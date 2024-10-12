# -*- coding: utf-8 -*-
"""
@author: agarc
"""

import pandas as pd
import os
from tqdm import tqdm
import numpy as np

# Setup
# Create Full Path - This is the OS agnostic way of doing so
dir_name = os.getcwd()
full_path_data = os.path.join(dir_name, 'data')
filename = 'SPX_HistoricalData.csv'
full_path = os.path.join(full_path_data, filename)
#
# Create the Main Data Frame
#
df_main = pd.read_csv(full_path) # read Excel spreadsheet
df_main.name = 'df_main' # name it
df_main['Date'] = pd.to_datetime(df_main['Date']) # Convert to Datetime
df_main = df_main.drop(columns = ['Volume']) # Drop Volume


# - Create Data Frames for the other Features -
# COMP (NASDAQ) Index
NASDAQ_df = pd.read_csv(os.path.join(full_path_data, 'NDAQ.csv'))
NASDAQ_df['Date'] = pd.to_datetime(NASDAQ_df['Date']) # Convert to Datetime
NASDAQ_df = NASDAQ_df.drop(columns = ['Volume']) # Drop Volume
NASDAQ_df.name = "NASDAQ" # name it

# NYSE Index
NYSE_df = pd.read_csv(os.path.join(full_path_data, 'NYSE_HistoricalData.csv'))
NYSE_df['Date'] = pd.to_datetime(NYSE_df['Date'])
NYSE_df = NYSE_df.drop(columns = ['Volume'])
NYSE_df.name = 'NYSE'

# Boing
boeing_df = pd.read_csv(os.path.join(full_path_data, 'boeing_STOCK_US_XNYS_BA_OCT17_OCT22.csv'))
boeing_df['Date'] = pd.to_datetime(boeing_df['Date'])
boeing_df = boeing_df.drop(columns = ['Volume'])
boeing_df.name = 'boeing'

# Oil
oil_df = pd.read_csv(os.path.join(full_path_data, 'Oil_Brent_10_31_22-10_02_17.csv'))
oil_df['Date'] = pd.to_datetime(oil_df['Date'])
oil_df = oil_df.drop(columns = ['Volume'])
oil_df.name = 'oil'

# Silicon
silicon_df = pd.read_csv(os.path.join(full_path_data, 'HistoricalPrices_silicon.csv'))
silicon_df['Date'] = pd.to_datetime(silicon_df['Date'])
#silicon_df = silicon_df.drop(columns = ['Volume'])
silicon_df.name = 'silicon'

# steel
steel_df = pd.read_csv(os.path.join(full_path_data, 'HistoricalPrices_steel.csv'))
steel_df['Date'] = pd.to_datetime(steel_df['Date'])
#steel_df = steel_df.drop(columns = ['Volume'])
steel_df.name = 'steel'

# Bitcoin
bitcoin_df = pd.read_csv(os.path.join(full_path_data, 'BTC-USD_OCT17_OCT22.csv'))
bitcoin_df['Date'] = pd.to_datetime(bitcoin_df['Date'])
bitcoin_df = bitcoin_df.drop(columns = ['Volume'])
bitcoin_df.name = 'bitcoin'

# Inflation
inflation_df = pd.read_excel(os.path.join(full_path_data, 'Inflation_rate_per_year.xlsx'))
inflation_df = inflation_df[inflation_df['Country Name'] == 'United States']
inflation_df = inflation_df.reset_index(drop = True)
inflation_df = inflation_df.drop(columns = ['Country Name','Country Code', 'Indicator Name', 'Indicator Code'])
inflation_df.name = 'inflation'


# Combine Stock Data to df_main
item_list = [NASDAQ_df, NYSE_df, boeing_df, oil_df, silicon_df, steel_df, bitcoin_df]

for df in tqdm(item_list):
    for idx in df_main.index:
        this_series = df[df['Date'] == df_main['Date'][idx]]
        for this_column in this_series.columns:
            try:
                df_main.loc[idx,str(df.name)+'_'+str(this_column)] = this_series[this_column].values[0]
            except:
                df_main.loc[idx,str(df.name)+'_'+str(this_column)] = np.NAN

# Drop Dates from Features

for item in item_list:
    df_main = df_main.drop(columns = [item.name+'_Date'])


#%%
# Delete missing rows
from  datetime import datetime
start_date = datetime.strptime("09.10.2017 00:00:00", '%d.%m.%Y %H:%M:%S')
end_date = datetime.strptime("03.10.2022 00:00:00", '%d.%m.%Y %H:%M:%S')

df_new = df_main.copy()

for i in tqdm(df_main.index):
    this_date = df_main.iloc[i]['Date']
    if this_date < start_date:
        df_new = df_new.drop([i])
    elif this_date > end_date:
        df_new = df_new.drop([i])


df_main = df_new.reset_index(drop=True)

# Generating a Report for RAW
from stats_report import StatsReport

labels = df_main.columns
report = StatsReport()

# Create a simple data set summary for the console
for thisLabel in tqdm(labels): # for each column, report stats
    thisCol = df_main[thisLabel]
    report.addCol(thisLabel, thisCol)

#print(report.to_string())
report.statsdf.to_excel("Quality_Report_Before_Prep.xlsx")

#%%
# Replace all Missing Values then Make Report - Should have no missing values
labels = df_main.columns
report = StatsReport()

for this_label in labels:
    df_main[this_label].fillna(df_main[this_label].mean(), inplace = True)

# Create a simple data set summary for the console
for thisLabel in tqdm(labels): # for each column, report stats
    thisCol = df_main[thisLabel]
    report.addCol(thisLabel, thisCol)

report.statsdf.to_excel("Quality_Report.xlsx")
 #%%
# Add Several Day Data

labels = df_main.columns
for this_label in labels:
    print(f'\nUsing Label: {this_label}')
    for idx in tqdm(df_main.index):
        if idx < 3 or idx > len(df_main):
            pass
        else:
            df_main.loc[idx, this_label + '_in_1_day'] = df_main.loc[idx-1][this_label]
            df_main.loc[idx, this_label + '_in_2_days'] = df_main.loc[idx-2][this_label]
            df_main.loc[idx, this_label + '_in_3_days'] = df_main.loc[idx-3][this_label]
#%%
# Add Commodities and Inflation
'''
for c_idx in commodity_df.index:
    for df_idx in tqdm(df_main.index):
        if commodity_df['Month'][c_idx].month == df_main['Date'][df_idx].month:
            #print(f'\nAdding Here: c_idx = {c_idx}, df_idx = {df_idx}')
            df_main.loc[df_idx,str(commodity_df.name)+'_'+str('Price')] = commodity_df['Price'][c_idx]
            df_main.loc[df_idx,str(commodity_df.name)+'_'+str('Change')] = commodity_df['Change'][c_idx]
        else:
            #print("Didn't Work")
            pass
'''        
for year in inflation_df.columns:
    for df_idx in tqdm(df_main.index):
        if year == df_main['Date'][df_idx].year:
            print('\nworked')
            df_main.loc[df_idx,'Inflation_Rate'] = inflation_df[year][0]

# Replace Inflation Rate of 2022 with Avg
df_main['Inflation_Rate'].fillna(df_main['Inflation_Rate'].mean(), inplace = True)

# Create Target Value - Stock Price 28 days later
import datetime

start = df_main['Date'][0] - datetime.timedelta(days = 84)
idx = df_main[df_main['Date'] == start].index[0]

for this_idx in tqdm(range(idx,len(df_main))):
    future_date = df_main['Date'] == df_main['Date'][this_idx] + datetime.timedelta(days = 84)
    if future_date.sum() == 0:
        before = df_main.loc[this_idx-1, 'Close/Last_in_84_Days']
        after = df_main.loc[this_idx-1, 'Close/Last_in_84_Days']
        avg = np.mean([before, after])
        df_main.loc[this_idx, 'Close/Last_in_84_Days'] = avg
    else:
        future_idx = df_main[future_date].index[0]
        df_main.loc[this_idx, 'Close/Last_in_84_Days'] = df_main['Close/Last'][future_idx]

df_main = df_main.drop(index = range(0,idx)).reset_index(drop = True)

#%%
# Assuming I buy GM stock every time model says yes to profit in 28 days
# Assuming over 3 Months period
def calculate_income(clf, X, df, investment):
    predY = clf.predict(X)
    total_income = 0
    for current_price, pred_future_price, actual_future_price, date in zip(df['Close/Last'], predY, df['Close/Last_in_84_Days'], df['Date'] ):
        if current_price < pred_future_price:
            income = (investment/current_price)*actual_future_price - investment
            #print(f'Made {income} on {date}')
        else:
            income = 0
        total_income = total_income+income
    return total_income

#%% Imports
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from itertools import product
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

#%% Target Data
# Setting up Training Data
# Data
predictors = df_main.drop(columns = ['Close/Last_in_84_Days']).columns
X = df_main[predictors].to_numpy(np.float64)
min_max_scaler = MinMaxScaler()
X_norm = min_max_scaler.fit_transform(X)

Y = df_main['Close/Last_in_84_Days'].to_numpy(np.float64)

# Split Testing and Training Data
X_train, X_test, y_train, y_test = train_test_split(X_norm, Y, test_size=0.3, 
                                                  train_size=0.7, random_state=22222, 
                                                  shuffle=True, stratify=None) 

#%%
# Linear Regression
linreg_model = LinearRegression().fit(X_train, y_train)
y_pred = linreg_model.predict(X_test)
print(f'Linear Regression Score: {linreg_model.score(X_test, y_test):.4f}')
print(f'Mean squared error (MSE): {mean_squared_error(y_test, y_pred):.4f}')
#print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred)}")

print(f"Income: {calculate_income(linreg_model, X_norm, df_main, 1000)}")


#%% LINEAR
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
    
f,ax=plt.subplots(nrows = 3, ncols = 2)
f.set_figwidth(24)
f.set_figheight(120)
f.suptitle('Actual and Predictions of Close Stock Price in 84 Days for Past 5 Years \nModel: SVR Radial Base Function', fontsize = 36)
c_values = [1,10,100,1000, 10000, 20000]
ax = ax.flatten()
for col, c in zip(ax, c_values):
    regr = make_pipeline(StandardScaler(), SVR(kernel = 'linear', C=c, epsilon=0.2))
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    predY = regr.predict(X_norm)
    
    print(f'SVR rfb Score: {regr.score(X_test, y_test):.4f}')
    print(f'Mean squared error (MSE): {mean_squared_error(y_test, y_pred):.4f}')
    #print(f"Accuracy: {r2_score(y_test, y_pred)}")
    print(f"Income: {calculate_income(regr, X_norm, df_main, 1000)}")

    col.plot(df_main['Date'].values, Y, label = 'Future Stock Price')
    col.plot(df_main['Date'].values, predY,label = 'Model Prediction')
    col.plot(df_main['Date'].values, df_main['Close/Last'], label = 'Current Stock Price')
    col.set_xlabel('Date', fontsize = 16)
    col.set_ylabel('SPX Stock Price at Close (USD)', fontsize = 16)
    col.set_title(f'Regularization Factor: {1/c}', fontsize = 24)
    col.legend(fontsize = 16)
    col.grid('on')

plt.show()
    

#%%  Radial Base Function
f,ax=plt.subplots(nrows = 3, ncols = 2)
f.set_figwidth(24)
f.set_figheight(120)
f.suptitle('Actual and Predictions of Close Stock Price in 84 Days for Past 5 Years \nModel: SVR Radial Base Function', fontsize = 36)
c_values = [1,10,100,1000, 10000, 20000]
ax = ax.flatten()
for col, c in zip(ax, c_values):
    regr = make_pipeline(StandardScaler(), SVR(kernel = 'rbf', C=c, epsilon=0.2))
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    predY = regr.predict(X_norm)
    
    print(f'SVR rfb Score: {regr.score(X_test, y_test):.4f}')
    print(f'Mean squared error (MSE): {mean_squared_error(y_test, y_pred):.4f}')
    #print(f"Accuracy: {r2_score(y_test, y_pred)}")
    print(f"Income: {calculate_income(regr, X_norm, df_main, 1000)}")

    col.plot(df_main['Date'].values, Y, label = 'Future Stock Price')
    col.plot(df_main['Date'].values, predY,label = 'Model Prediction')
    col.plot(df_main['Date'].values, df_main['Close/Last'], label = 'Current Stock Price')
    col.set_xlabel('Date', fontsize = 16)
    col.set_ylabel('SPX Stock Price at Close (USD)', fontsize = 16)
    col.set_title(f'Regularization Factor: {1/c}', fontsize = 24)
    col.legend(fontsize = 16)
    col.grid('on')

plt.show()

    




#%% Polynomial
f,ax=plt.subplots(nrows = 3, ncols = 2)
f.set_figwidth(24)
f.set_figheight(120)
f.suptitle('Actual and Predictions of Close Stock Price in 84 Days for Past 5 Years \nModel: SVR Polynomial', fontsize = 36)
c_values = [1,10,100,1000, 10000, 20000]
ax = ax.flatten()
for col, c in zip(ax, c_values):
    regr = make_pipeline(StandardScaler(), SVR(kernel = 'poly', C=c, epsilon=0.2))
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    predY = regr.predict(X_norm)
    
    print(f'SVR polynomial Score: {regr.score(X_test, y_test):.4f}')
    print(f'Mean squared error (MSE): {mean_squared_error(y_test, y_pred):.4f}')
    #print(f"Accuracy: {r2_score(y_test, y_pred)}")
    print(f"Income: {calculate_income(regr, X_norm, df_main, 1000)}")
    
    # Plot 5 Years
    col.plot(df_main['Date'].values, Y, label = 'Future Stock Price')
    col.plot(df_main['Date'].values, predY,label = 'Model Prediction')
    col.plot(df_main['Date'].values, df_main['Close/Last'], label = 'Current Stock Price')
    col.set_xlabel('Date', fontsize = 16)
    col.set_ylabel('SPX Stock Price at Close (USD)', fontsize = 16)
    col.set_title(f'Regularization Factor: {1/c}', fontsize = 24)
    col.legend(fontsize = 16)
    col.grid('on')
plt.show()



















# ARCHIVE #




#%% Create Binary Target DataFrame
df_bin_target = df_main.copy()
for idx in df_bin_target.index:
    if df_bin_target.loc[idx, 'GM_Close/Last'] < df_bin_target.loc[idx, 'GM_Close/Last_in_28_Days']:
        df_bin_target.loc[idx, 'Profit'] = True
    else:
        df_bin_target.loc[idx, 'Profit'] = False
        
# Setting up Training Data
predictors = df_bin_target.drop(columns = ['Profit']).columns
X_bin = df_bin_target[predictors].to_numpy(np.float64)
min_max_scaler = MinMaxScaler()
X_bin_norm = min_max_scaler.fit_transform(X_bin)

Y_bin = df_bin_target['Profit'].to_numpy(np.float64)

# Split Testing and Training Data
X_bin_train, X_bin_test, y_bin_train, y_bin_test = train_test_split(X_bin_norm, Y_bin, test_size=0.3, 
                                                                    train_size=0.7, random_state=22222, 
                                                                    shuffle=True, stratify=None)
#%% Logistic Regression
modelLog = LogisticRegression()
clf_Log = modelLog.fit(X_bin_train, y_bin_train)
Ypred = clf_Log.predict(X_bin_test)
Ypredclass = 1*(Ypred > 0.5)
print("Logistic Regression \nR2 = %f,  MSE = %f,  Classification Accuracy = %f" % (metrics.r2_score(y_bin_test, Ypred), metrics.mean_squared_error(y_bin_test, Ypred), metrics.accuracy_score(y_bin_test, Ypredclass)))

Ypred = clf_Log.predict(X_bin_norm)
total_income = 0
for current_price, pred, actual_future_price in zip(df_main['GM_Close/Last'], Ypred, df_main['GM_Close/Last_in_28_Days'] ):
    if pred == 1:
        income = (1000/current_price)*actual_future_price - 1000
    else:
        income = 0
    total_income = total_income+income

print(f'Income: {total_income}')
#%% - MIGHT NOT NEED - DELETE LATER
'''
# Poly
poly = PolynomialFeatures(2) # object to generate polynomial basis functions
#X1_train = df_main.drop([targetName], axis=1).to_numpy()
bigTrainX = poly.fit_transform(X1_train)
mlrf = LogisticRegression() # creates the regressor object
mlrf.fit(bigTrainX, y_train)
Ypred1 = mlrf.predict(bigTrainX)
Ypredclass1 = 1*(Ypred1 > 0.5)
print("R2 = %f, MSE = %f, Classification Accuracy = %f" % (metrics.r2_score(y1_test, Ypred1), metrics.mean_squared_error(y1_test, Ypred1), metrics.accuracy_score(y1_test, Ypredclass1)))
print("W: ", np.append(np.array(mlrf.intercept_), mlrf.coef_))
'''
#%% - Neural Network
from sklearn.neural_network import MLPRegressor

from datetime import datetime
t1 = datetime.now()
resultsDF = pd.DataFrame(data=None, columns=['Hidden Layer', 'Activation', 'Mean Squared Error', 'Train Accuracy', 'Test Accuracy'])
#firstHL = list(range(150,200))
#secondHL = list(product(range(100,200),repeat=2))

neurons = [10,50,100,150,200]
thirdHL = list(product(neurons,repeat=3))
hiddenLayers = thirdHL

activations = ('relu', 'logistic', 'identity', 'tanh')
# Hidden Layers
for act in activations:
    for hl in hiddenLayers:
        #hl = i
        #currActivate = k
        #regpenalty = 0.0003 #according to forums on the internet, this is an optimal adam solver learning rate
        clf = MLPRegressor(hidden_layer_sizes=(hl)
                            , activation=act
                            , solver='adam'
                            , alpha=0.04
                            , max_iter=10000
                            , validation_fraction=0.42).fit(X_train,y_train)
        annPredY = clf.predict(X_test)
        
        # Get  Scores
        train_accuracy = clf.score(X_train, y_train)
        test_accuracy = clf.score(X_test, y_test)
        mse = metrics.mean_squared_error(y_test, annPredY)
        
        print("\n ###### MPL Classifier #######")
        print(f"\n Activation Type: {act}")
        #print(f"\nLearning Rate: {learning_rate}")
        print(f"\nHidden Layers: {hl}")
        print("\n\rANN: MSE = %f" % mse)
        print(f"\nTrain Accuracy = {train_accuracy}")
        print(f"\nTest Accuracy = {test_accuracy}")
   
        resultsDF = resultsDF.append({'Hidden Layer': hl, 
                                      'Activation': act, 
                                      'Mean Squared Error': mse,
                                      'Train Accuracy': train_accuracy,
                                      'Test Accuracy':test_accuracy}, ignore_index=True)

t2 = datetime.now()
total_time = t2-t1
print("Started: ", t1)
print("Ended: ", t2)
print("Total Time for ANN: ", t2-t1)


#%% - Save ANN DF
best_test = resultsDF['Test Accuracy'].nlargest(10)
#best_misslabeled = resultsDF['misclass'].nsmallest(10)
worst_test = resultsDF['Test Accuracy'].nsmallest(10)
#worst_mislabeled = resultsDF['misclass'].nlargest(10)

best_test_df = pd.DataFrame([resultsDF.loc[best_test.index[idx]] for idx in range(len(best_test.index))])
#best_misslabeled_df = pd.DataFrame([resultsDF.loc[best_misslabeled.index[idx]] for idx in range(len(best_misslabeled.index))])
worst_test_df = pd.DataFrame([resultsDF.loc[worst_test.index[idx]] for idx in range(len(worst_test.index))])
#worst_mislabeled_df = pd.DataFrame([resultsDF.loc[worst_mislabeled.index[idx]] for idx in range(len(worst_mislabeled.index))])

best_test_df.to_excel('Best_Test_Accuracy.xlsx')
#best_misslabeled_df.to_excel('Best_Mislabeled_Models.xlsx')
worst_test_df.to_excel('Worst_Test_Accuracy.xlsx')
#worst_mislabeled_df.to_excel('Worst_Mislabeled_Models.xlsx')


#%%
# Ensemble (TY SCIKIT LEARN DOCUMENTATION!)
# Random Forest Bc That Sounds The Most Useful (Did I Profit?)

from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor

X, y = make_blobs(n_samples=10000, n_features=10, centers=100,random_state=0)

clf = DecisionTreeRegressor(max_depth=None, min_samples_split=2,random_state=0).fit(X_train, y_train)
predY = clf.predict(X_test)
scores = cross_val_score(clf, X_test, y_test, cv=5)
print(scores.mean())
print(f'Decision Tree \nR2 Score: {metrics.r2_score(y_test, predY)}')
print(f'MSE: {metrics.mean_squared_error(y_test, predY)}')
print(f'Income: {calculate_income(clf, X_norm, df_main, 1000)}')

clf = RandomForestRegressor(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0).fit(X_train, y_train)
predY = clf.predict(X_test)
scores = cross_val_score(clf, X_test, y_test, cv=5)
print(scores.mean())
print(f'Random Forest \nR2 Score: {metrics.r2_score(y_test, predY)}')
print(f'MSE: {metrics.mean_squared_error(y_test, predY)}')
print(f'Income: {calculate_income(clf,X_norm, df_main, 1000)}')

clf = ExtraTreesRegressor(n_estimators=10, max_depth=None,
                           min_samples_split=2, random_state=0).fit(X_train, y_train)
predY = clf.predict(X_test)
scores = cross_val_score(clf, X_test, y_test, cv=5)


print(scores.mean())
print(f'Extra Tree \nR2 Score: {metrics.r2_score(y_test, predY)}')
print(f'MSE: {metrics.mean_squared_error(y_test, predY)}')
print(f'Income: {calculate_income(clf,X_norm, df_main, 10000)}')
#%% Create ANN Best
clf_best = MLPRegressor(hidden_layer_sizes=(200,150,10)
                        , activation='tanh'
                        , solver='adam'
                        , alpha=0.04
                        , max_iter=10000
                        , validation_fraction=0.42).fit(X_train,y_train)
clf_best = clf_best.fit(X_train, y_train.ravel())
annPredY = clf_best.predict(X_norm)

clf_best.predict(X_test)
print(f'R2 Score for ANN: {metrics.r2_score(y_test, clf_best.predict(X_test))}')
print(f'Income: {calculate_income(clf_best,X_norm, df_main)}')
#%%
import matplotlib.pyplot as plt

clf = ExtraTreesRegressor(n_estimators=10, max_depth=None,
                           min_samples_split=2, random_state=0).fit(X_train, y_train)
predY = clf.predict(X_norm)

    
#%%
# Plot 5 Years
plt.figure()
plt.scatter(df_main['Date'].values, Y, label = 'Future Stock Price')
plt.plot(df_main['Date'].values, annPredY,label = 'Model Prediction')
plt.plot(df_main['Date'].values, df_main['GM_Close/Last'], label = 'Current Stock Price')
plt.xlabel('Date')
plt.ylabel('GM Stock Price at Close (USD)')
plt.title('Actual and Predictions of Close Stock Price in 28 Days for Past 5 Years \nModel: Extra Trees Regressor')
plt.legend()
plt.show()

## - 3 Months
df_3months = df_main.drop(index = range(63, len(df_main)))
#predictors = df_3months.drop(columns = ['GM_Close/Last_in_28_Days']).columns
#X_3months = df_3months[predictors].to_numpy(np.float64)
#min_max_scaler = MinMaxScaler()
#X_norm_3months = min_max_scaler.fit_transform(X_3months)
#Y_3months = df_3months['GM_Close/Last_in_28_Days'].to_numpy(np.float64)
#print(Y_3months)

# Prediction
Y_3months = Y[:63]
annPredY_3 = clf_best.predict(X_norm[:63])
print(annPredY_3)

# Plot 3 Months
plt.figure()
plt.plot(df_3months['Date'].values, Y_3months, label = 'Future Stock Price')
plt.plot(df_3months['Date'].values, annPredY_3, label = 'Model Prediction')
plt.plot(df_3months['Date'].values, df_main['GM_Close/Last'][:63], label = 'Current Stock Price')
plt.xlabel('Date')
plt.ylabel('GM Stock Price at Close (USD)')
plt.title('Actual and Predictions of Close Stock Price in 28 Days for Past 3 Months \nModel: Extra Trees Regressor')
plt.legend()
plt.show()


