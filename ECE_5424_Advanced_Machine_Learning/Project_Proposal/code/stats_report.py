# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 00:13:18 2022

@author: agarc
"""

import pandas

class StatsReport:
    def __init__(self):
        self.statsdf = pandas.DataFrame()
        self.statsdf['stat'] = ['cardinality', 'mean', 'median', 'n_at_median', 'mode', 'n_at_mode', 'stddev', 'min', 'max', 'nzero', 'nmissing']
        pass
    
    def addCol(self, label, data):
        self.statsdf[label] = [self.cardinality_(data), self.mean_(data), self.median_(data), self.n_at_median(data), self.mode_(data), self.n_at_mode(data), self.std_(data), self.min_(data), self.max_(data), self.nzero_(data), self.nmissing_(data)]
        
    def to_string(self):
        return self.statsdf.to_string()
    
    def cardinality_(self, d):
        try:
            return d.nunique()
        except:
            return "N/A"
        
    def mean_(self, d):
        try:
            return d.mean()
        except:
            return "N/A"

    def median_(self, d):
        try:
            return d.median()
        except:
            return "N/A"

    def n_at_median(self, d):
        try:
            n = d == d.median()
            return n.sum()
        except:
            return "N/A"     
        
    def mode_(self, d):
        try:
            return int(d.mode())
        except:
            return "N/A"      
        
    def n_at_mode(self, d):
        try:
            n = d == int(d.mode())
            return n.sum()
        except:
            return "N/A"
        
    def std_(self, d):
        try:
            return d.std()
        except:
            return "N/A"     
        
    def min_(self, d):
        try:
            return d.min()
        except:
            return "N/A"       
        
    def max_(self, d):
        try:
            return d.max()
        except:
            return "N/A"
        
    def nzero_(self, d):
        try:
            n = d == 0
            return n.sum()
        except:
            return "N/A"      
        
    def nmissing_(self, d):
        try:
            n = d.isna()
            return n.sum()
        except:
            return "N/A"
