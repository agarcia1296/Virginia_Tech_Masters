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
    
    def addCol(self, label, d):
        self.statsdf[label] = [d.nunique(), d.mean(), d.median(), d.median(), d.mode(), d.mode(), d.std(), d.min(), d.max(), d.nzero(), d.nmissing()]
    
    def to_string(self):
        return self.statsdf.to_string()