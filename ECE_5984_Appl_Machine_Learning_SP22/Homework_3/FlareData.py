# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 00:57:49 2022

@author: agarc
"""

import pandas as pd
import os
from sklearn import tree
import pydotplus 
import collections 
import stats_report as sr
#%% Functions

# for a two-class tree, call this function like this: 
# writegraphtofile(clf, ('F', 'T'), dirname+graphfilename) 
def writegraphtofile(clf, feature_labels, classnames, pathname): 
    dot_data = tree.export_graphviz(clf, out_file=None, 
                                    feature_names=feature_labels, 
                                    class_names=classnames, 
                                    filled=True, rounded=True, 
                                    special_characters=True) 
    graph = pydotplus.graph_from_dot_data(dot_data) 
    colors = ('lightblue', 'green') 
    edges = collections.defaultdict(list) 
    for edge in graph.get_edge_list(): 
        edges[edge.get_source()].append(int(edge.get_destination())) 
    for edge in edges: 
        edges[edge].sort() 
        for i in range(2): 
            dest = graph.get_node(str(edges[edge][i]))[0] 
            dest.set_fillcolor(colors[i])
    graph.write_png(pathname)
        
#%% Setup
# Create Full Path - This is the OS agnostic way of doing so
dir_name = os.getcwd()
filename = 'FlareData.xlsx'
full_path = os.path.join(dir_name, filename)

#
# Create the Data Frame
#
df = pd.read_excel(full_path) # read Excel spreadsheet
print('File {0} is of size {1}'.format(full_path, df.shape))
labels = df.columns
#feature_labels = labels.drop(["C class", "M class", "X class"])

#features = df[feature_labels]
#target = df["C class"]

#%% Simple Stats
#
# Getting Simple Stats based on HW1
#
report = sr.StatsReport()

# Create a simple data set summary for the console
for thisLabel in labels: # for each column, report stats
    thisCol = df[thisLabel]
    report.addCol(thisLabel, thisCol)
    
print(report.to_string())
#report.statsdf.to_excel("FlareData_Report.xlsx")

#%% Preprocessing
from sklearn.preprocessing import OrdinalEncoder

# Identify the Unique values of Ordianal Data
zurich_class_unique = pd.DataFrame(df["Zurich Class"].unique())
spot_size_unique = pd.DataFrame(df["Spot Size"].unique())
spot_dist_unique = pd.DataFrame(df["Spot Dist"].unique())

# Encode the Ordinal Data
encoder = OrdinalEncoder()
zur_class_encoded = pd.DataFrame(encoder.fit_transform(zurich_class_unique))
spot_size_encoded = pd.DataFrame(encoder.fit_transform(spot_size_unique))
spot_dist_encoded = pd.DataFrame(encoder.fit_transform(spot_dist_unique))

df_encoded = df.copy()

for idx in range(len(zur_class_encoded)):
    old_value = zurich_class_unique[0][idx]
    new_value = int(zur_class_encoded[0][idx])
    df_encoded["Zurich Class"] = df_encoded["Zurich Class"].replace(old_value,int(new_value))
    
for idx in range(len(spot_size_encoded)):
    old_value = spot_size_unique[0][idx]
    new_value = spot_size_encoded[0][idx]
    df_encoded["Spot Size"] = df_encoded["Spot Size"].replace(old_value,int(new_value))
    
for idx in range(len(spot_dist_encoded)):
    old_value = spot_dist_unique[0][idx]
    new_value = spot_dist_encoded[0][idx]
    df_encoded["Spot Dist"] = df_encoded["Spot Dist"].replace(old_value,int(new_value))
 
# Combining the new and old values into one
zur_class_encoded['Old Values'] = zurich_class_unique
spot_size_encoded['Old Values'] = spot_size_unique
spot_dist_encoded['Old Values'] = spot_dist_unique

print(f"[DATA after Ordinals are Encoded] \n{df_encoded}")
print(f"[Zurich Class encoder key] \n{zur_class_encoded}")
print(f"[Spot Size encoder key] \n{spot_size_encoded}")
print(f"[Spot Dist encoder key] \n{spot_dist_encoded}")

target = df_encoded["C class"]
target_unique = target.unique()

labels = df_encoded.columns
feature_labels = labels.drop(["C class", "M class", "X class"])
features = df_encoded[feature_labels]
#%% One Hot Encoding
#enc = OneHotEncoder(categories = "auto", drop = None, sparse = False, handle_unknown = 'error')
#categorical = df.loc[:,["Zurich Class", "Spot Size", "Spot Dist"]]
#enc_catagorical = enc.fit_transform(categorical)
#column_trans = make_column_transformer(
#    (OneHotEncoder(), ["Zurich Class", "Spot Size", "Spot Dist"]),
#    remainder = 'passthrough')
#df_new = column_trans.fit_transform(df)

#%% Decision Tree Entropy

clf_entropy = tree.DecisionTreeClassifier(criterion = "entropy", max_depth = 4)
clf_entropy = clf_entropy.fit(features, target)
print("Training set score = ", clf_entropy.score(features, target))
#print("Test set score = ", clf.score(testX, testy))

path_name = os.path.join(dir_name, "FlareData_DecisionTree_Entropy.png")
writegraphtofile(clf_entropy, feature_labels, (str(target_unique[0]), str(target_unique[1]), str(target_unique[2])), path_name)
tree.export_graphviz(clf_entropy)

#%% Decision Tree Entropy

clf_gini = tree.DecisionTreeClassifier(criterion = "gini", max_depth = 4)
clf_gini = clf_gini.fit(features, target)
print("Training set score = ", clf_gini.score(features, target))
#print("Test set score = ", clf.score(testX, testy))

path_name = os.path.join(dir_name, "FlareData_DecisionTree_Gini.png")
writegraphtofile(clf_gini, feature_labels, (str(target_unique[0]), str(target_unique[1]), str(target_unique[2])), path_name)
tree.export_graphviz(clf_gini)

