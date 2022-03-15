# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 22:20:25 2022

@author: agarc
"""

import pandas as pd
import os
from tqdm import tqdm
#%% Setup
# Create Full Path - This is the OS agnostic way of doing so
dir_name = os.getcwd()
filename = 'USW00023066.csv'
full_path = os.path.join(dir_name, filename, filename)

#
# Create the Main Data Frame
#
data_headers = ['ID', 'DATE', 'ELEMENT', 'VALUE1', 'MFLAG1', 'Q_FLAG1', 'SFLAG1', 'VALUE2']
df_main = pd.read_csv(full_path, names = data_headers) # read Excel spreadsheet
print('File {0} is of size {1}'.format(full_path, df_main.shape))


#%% Generating a Report for RAW
from utils_project1 import StatsReport

labels = df_main.columns
report = StatsReport()

# Create a simple data set summary for the console
for thisLabel in tqdm(labels): # for each column, report stats
    thisCol = df_main[thisLabel]
    report.addCol(thisLabel, thisCol)

print(report.to_string())
report.statsdf.to_excel("Report_Project1_RAW_DATA.xlsx")

#%%
def get_unique_column_values(df):
    """
    Identifying Unique Values of each Column in DF
    Output is a Dictionary of each Column
    """
    headers_unique = {}
    for label in tqdm(df.columns):
        headers_unique[label] = df[label].unique()
    #pbar.close()
    return headers_unique

headers_unique = get_unique_column_values(df_main)
print(f"List of Dates: {headers_unique['DATE']}")

#%% Data Preperation - THIS TAKES SEVERAL MINUTES

def prep_data(df_main, df_prep, headers_unique):    
    """
    Extract Values for Elements and insert into df_prep
    """
    index_ = 0 
    for date in tqdm(headers_unique['DATE']):
        date_idx = df_main['DATE'] == date
        df_by_date = df_main[date_idx]
        df_prep.loc[index_, 'DATE'] = date 
        for idx in df_by_date['ELEMENT'].index:
            df_prep.loc[index_, df_by_date['ELEMENT'][idx]] = df_by_date['VALUE1'][idx]
        index_ = index_+1
    

df_prep = pd.DataFrame(columns = ['DATE', *headers_unique['ELEMENT']])
prep_data(df_main, df_prep, headers_unique)

df_ex = df_main.head(10)
df_prep_ = pd.DataFrame(columns = ['DATE', *headers_unique['ELEMENT']])
prep_data(df_ex, df_prep_, headers_unique)
#%%
#
# Create Columns - PRECIPFLAG and PRECIPAMT 
# Create Target Columns - NEXTDAYPRECIPFLAG and NEXTDAYPRECIPAMT
#
for idx in tqdm(df_prep.index):
    rain = df_prep['PRCP'][idx] # in tenths of mm
    snow = df_prep['SNOW'][idx]
    if (rain or snow) > 0:
        df_prep.loc[idx, 'PRECIPFLAG'] = 1 # It rained/snowed
        df_prep.loc[idx, 'PRECIPAMT'] = 0.0393701*(rain/10) + (0.0393701*snow)/8 # result is in inches
    else:
        df_prep.loc[idx, 'PRECIPFLAG'] = 0 # It did not rain/snow
        df_prep.loc[idx, 'PRECIPAMT'] = 0
    if idx > 0:
        df_prep.loc[idx-1, 'NEXTDAYPRECIPFLAG'] = df_prep.loc[idx, 'PRECIPFLAG']
        df_prep.loc[idx-1, 'NEXTDAYPRECIPAMT'] = df_prep.loc[idx, 'PRECIPAMT']

# Output to Excel
df_prep.to_excel('project1_prepped.xlsx')

#%% Generating a Report
labels = df_prep.columns
report = StatsReport()

# Create a simple data set summary for the console
for thisLabel in tqdm(labels): # for each column, report stats
    thisCol = df_prep[thisLabel]
    report.addCol(thisLabel, thisCol)

#print(report.to_string())
report.statsdf.to_excel("Report_Project1_post.xlsx")

#%% Setting up Training Data
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import numpy as np
# Data
X_labels = df_prep.columns.drop(['NEXTDAYPRECIPFLAG','NEXTDAYPRECIPAMT'])
X = df_prep[X_labels]
# Target
y = df_prep.loc[:, ['NEXTDAYPRECIPFLAG','NEXTDAYPRECIPAMT']]

# 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7,random_state=1, shuffle=True, stratify=None)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

print(np.mean(X_train))
print(np.mean(X_test))
print(np.mean(y_train))
print(np.mean(y_test))
#%% Decidion Tree - Prediction of Rain on Next Day
from utils_project1 import writegraphtofile
from sklearn import tree

X_train.dropna()
y_train.dropna()

clf_entropy = tree.DecisionTreeClassifier(criterion = "entropy", max_depth = 4)
clf_entropy = clf_entropy.fit(X_train, y_train)
print("Training set score = ", clf_entropy.score(X_train, y_train))
print("Test set score = ", clf_entropy.score(X_test, y_test))

path_name = os.path.join(dir_name, "FlareData_DecisionTree_Entropy.png")
writegraphtofile(clf_entropy, feature_labels, (str(target_unique[0]), str(target_unique[1]), str(target_unique[2])), path_name)
tree.export_graphviz(clf_entropy)













