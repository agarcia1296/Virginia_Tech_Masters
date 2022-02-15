# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 00:10:01 2022

@author: agarc
"""

import pandas
import stats_report as sr

filename = r"C:\Users\agarc\OneDrive\Documents\GitHub\Virginia_Tech_Masters\ECE_5984_Appl_Machine_Learning_SP22\Homework_2\Heart Disease.xlsx"
df = pandas.read_excel(filename) # read Excel spreadsheet
print('File {0} is of size {1}'.format(filename, df.shape))
labels = df.columns
report = sr.StatsReport()

# Create a simple data set summary for the console
for thisLabel in labels: # for each column, report stats
    thisCol = df[thisLabel]
    try:
        report.addCol(thisLabel, thisCol)
    except ValueError:
        pass
    print(report.to_string())