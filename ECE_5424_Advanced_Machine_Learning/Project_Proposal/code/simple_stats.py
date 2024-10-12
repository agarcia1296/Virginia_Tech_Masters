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

    report.addCol(thisLabel, thisCol)

print(report.to_string())
report.statsdf.to_excel("Report_Andrew_Garcia.xlsx")

covariance = df.cov()
correlation = df.corr()
covariance.to_excel("Covariance_Andrew_Garcia.xlsx")
correlation.to_excel("Correlation_Andrew_Garcia.xlsx")


labels = correlation.columns
for thisLabel in labels:
    if thisLabel == "member":    
        pass
    else:
        thisCol = correlation[thisLabel]
        v = thisCol.sort_values()
        max_v = v[-2]
        min_v = v[0]
        min_v_abs = abs(min_v)
        if max_v > min_v_abs:
            max_overall = max_v
            print(f"\n Label: {thisLabel} - Highest: {max_v}")
        elif min_v_abs > max_v:
            max_overall = min_v_abs
            print(f"\n Label: {thisLabel} - Highest: {min_v}")

    