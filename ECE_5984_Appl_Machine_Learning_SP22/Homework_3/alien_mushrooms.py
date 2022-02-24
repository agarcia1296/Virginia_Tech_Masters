# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 23:50:32 2022

@author: agarc
"""

import pandas
import os
import sklearn

# Create Full Path - This is the OS agnostic way of doing so
dir_name = os.getcwd()
filename = 'AlienMushrooms.xlsx'
full_path = os.path.join(dir_name, filename)

# Create the Data Frame
df = pandas.read_excel(full_path) # read Excel spreadsheet
print('File {0} is of size {1}'.format(full_path, df.shape))
labels = df.columns


import pydotplus 
import collections 
# for a two-class tree, call this function like this: 
# writegraphtofile(clf, ('F', 'T'), dirname+graphfilename) 
def writegraphtofile(classifier, classnames, pathname): 
    dot_data = classifier.export_graphviz(clf, out_file=None, 
                                    feature_names=featurelabels.tolist(), 
                                    class_names=classnames, 
                                    filled=True, rounded=True, 
                                    special_characters=True) 
    graph = pydotplus.graph_from_dot_data(dot_data) 
    colors = ('lightblue', 'lightgreen') 
    edges = collections.defaultdict(list) 
    for edge in graph.get_edge_list(): 
        edges[edge.get_source()].append(int(edge.get_destination())) 
    for edge in edges: 
        edges[edge].sort() 
        for i in range(2): 
            dest = graph.get_node(str(edges[edge][i]))[0] 
            dest.set_fillcolor(colors[i])
        graph.write_png(pathname)
        
        
def write_graph_to_file(classifier, classnames, pathname):
    dot_data = tree.export_graphviz(clf)