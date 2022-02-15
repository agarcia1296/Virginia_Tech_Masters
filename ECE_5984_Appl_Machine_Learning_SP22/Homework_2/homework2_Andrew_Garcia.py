"""
Created on Sat Feb 12 15:08:23 2022

@author: agarc
"""
import pandas
import numpy as np

filename = r"C:\Users\agarc\OneDrive\Documents\GitHub\Virginia_Tech_Masters\ECE_5984_Appl_Machine_Learning_SP22\Homework_2\Heart Disease.xlsx"
df = pandas.read_excel(filename) # read an Excel spreadsheet
print('File {0} is of size {1}'.format(filename, df.shape))

labels = df.columns
featureLabels = labels.drop('target').values # get just the predictors
xFrame = df[featureLabels]
yFrame = df['target'] # and the target variable
predictors = xFrame.to_numpy # convert them to numpy arrays
target = yFrame.to_numpy

# Create a simple data set summary for the console
for thisLabel in labels: # for each column, report basic stats
    thisCol = df[thisLabel]
    meanV = thisCol.mean()  
    minV = thisCol.min()
    maxV = thisCol.max()
    print('Col {0}: mean = {1}, min = {2}, max = {3}'.format(thisLabel, meanV, minV, maxV))