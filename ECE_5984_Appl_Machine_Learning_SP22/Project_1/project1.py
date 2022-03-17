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
full_path = os.path.join(dir_name, filename)

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
report.statsdf.to_excel("Quality_Report_Before_Prep.xlsx")

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

def prep_data(df, df_out, headers_unique):    
    """
    Extract Values for Elements and insert into df_prep
    """
    index_ = 0 
    for date in tqdm(headers_unique['DATE']):
        date_idx = df['DATE'] == date
        df_by_date = df[date_idx]
        df_out.loc[index_, 'DATE'] = date 
        for idx in df_by_date['ELEMENT'].index:
            df_out.loc[index_, df_by_date['ELEMENT'][idx]] = df_by_date['VALUE1'][idx]
        index_ = index_+1
    

df_prep = pd.DataFrame(columns = ['DATE', *headers_unique['ELEMENT']])
prep_data(df_main, df_prep, headers_unique)

#%% Create Target Columns
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

#%% Generating a Report
labels_post = df_prep.columns
report_post = StatsReport()

# Create a simple data set summary for the console
for thisLabel in tqdm(labels_post): # for each column, report stats
    thisCol = df_prep[thisLabel]
    report_post.addCol(thisLabel, thisCol)

#print(report.to_string())
report_post.statsdf.to_excel("Quality_Report_Post_Prep.xlsx")

#%% Sus out Bad Elements
from utils_project1 import replace_missing_values_avg

df_final = df_prep.copy()

temp_report_df = report_post.statsdf

for element in tqdm(labels_post):
    if temp_report_df[element][10] > len(df_prep)*0.1: # Weeding out Elements that have more than 10% of missing values
        df_final = df_final.drop(columns = [element])
    elif temp_report_df[element][10] < len(df_prep)*0.1:
        if element == 'NEXTDAYPRECIPAMT' or element == 'NEXTDAYPRECIPFLAG':
            avg_value = 0
        else:
            avg_value = temp_report_df[element][1]
        replace_missing_values_avg(df_final, element, avg_value)

#%%
# Run Quality Report and Output Data to Excel
df_final.to_excel('Weather_Data_Final.xlsx')

labels_final = df_final.columns
report_final = StatsReport()

# Create a simple data set summary for the console
for thisLabel in tqdm(labels_final): # for each column, report stats
    thisCol = df_final[thisLabel]
    report_final.addCol(thisLabel, thisCol)

#print(report.to_string())
report_final.statsdf.to_excel("Quality_Report_Final.xlsx")

#%% Setting up Training Data
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import numpy as np

# Data
X_labels = df_final.columns.drop(['NEXTDAYPRECIPFLAG','NEXTDAYPRECIPAMT'])
X = df_final[X_labels]

# Target
y_precip_flag = df_final.loc[:, ['NEXTDAYPRECIPFLAG']]
y_precip_amt = df_final.loc[:, ['NEXTDAYPRECIPAMT']]

#%% Decision Tree - Next Day Precip Flag
from utils_project1 import writegraphtofile
from sklearn import tree

# Create Testing and Training data
X_train, X_test, y_train, y_test = train_test_split(X, y_precip_flag, test_size=0.3, train_size=0.7, random_state=1, shuffle=True, stratify=None)

# Create Decision Tree
clf_entropy = tree.DecisionTreeClassifier(criterion = "entropy", max_depth = 4)
clf_entropy = clf_entropy.fit(X_train, np.array(y_train['NEXTDAYPRECIPFLAG']))
print("Training set score = ", clf_entropy.score(X_train, y_train['NEXTDAYPRECIPFLAG']))
print("Test set score = ", clf_entropy.score(X_test, y_test['NEXTDAYPRECIPFLAG']))

# Create Graphic
target_unique = y_precip_flag['NEXTDAYPRECIPFLAG'].unique()
path_name = os.path.join(dir_name, "Weather_Data_DecisionTree_Entropy.png")
writegraphtofile(clf_entropy, X_labels, (str(target_unique[0]), str(target_unique[1])), path_name)
tree.export_graphviz(clf_entropy)

#%% 


# Next Day Precip Amt
X_train, X_test, y_train, y_test = train_test_split(X, y_precip_amt, test_size=0.3, train_size=0.7,random_state=1, shuffle=True, stratify=None)
















