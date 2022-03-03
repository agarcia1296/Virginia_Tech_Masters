# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 21:53:30 2022

@author: agarc
"""

"""
TestID3 -Creed Jones created on Feb 12 2020
A simple exploration of decision tree classifiers
"""
from sklearn import tree# the decision tree functionality
import pydot# to write the tree as a pdf image
shirts = [[1,1,10], [1,2,10], [1,3,10], [2,1,9], [2,2,9], [2,3,9], [2,1,16],
          [2,2,16], [2,3,16], [3,1,10], [3,2,10], [3,3,10]]
F = [2, 3, 3, 1, 2, 2, 1, 1, 1, 1, 2, 2]
shirtNames= ["ColorIndex", "SizeIndex", "Price"]
targetNames= ["Low", "Medium", "High"]

clf= tree.DecisionTreeClassifier(criterion="entropy")# create a tree object to do classification
clf= clf.fit(shirts, F)# train it on this data

dot_data= tree.export_graphviz(clf, out_file=None, # merely to write the tree out
                               feature_names=shirtNames,
                               class_names=targetNames,
                               filled=True, rounded=True,
                               special_characters=True)

(graph,) = pydot.graph_from_dot_data(dot_data)
graph.write_png("shirtstree.png")