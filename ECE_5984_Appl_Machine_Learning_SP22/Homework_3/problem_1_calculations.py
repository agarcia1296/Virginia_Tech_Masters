# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 23:23:51 2022

@author: agarc
"""

import pandas
import os
import sklearn
import math

def entropy(target):
    # Works for any number of classes
    total = len(target)
    options = target.value_counts()
    sum_ = 0
    #
    for level in options:
        p = level/total
        this_value = p*math.log2(p)
        sum_ = sum_ + this_value
    return sum_

def rem_entropy (d_feature, target):
    # must be Booleon array
    total = len(target)
    true_count = d_feature.sum()
    false_count = total - true_count
    p_true = true_count/total
    p_false = false_count/total
    
    # get df for when true/false and target
    true_feature_at_target = target[d_feature]
    false_feature_at_target = target[~d_feature]
    
    # Calculate entropy
    true_entropy = entropy(true_feature_at_target)
    false_entropy = entropy(false_feature_at_target)
    
    final_sum = p_true*(-1*true_entropy) + p_false*(-1*false_entropy)
    return final_sum
#%%
# Create Full Path - This is the OS agnostic way of doing so
dir_name = os.getcwd()
filename = 'AlienMushrooms.xlsx'
full_path = os.path.join(dir_name, filename)

# Create the Data Frame
df = pandas.read_excel(full_path) # read Excel spreadsheet
print('File {0} is of size {1}'.format(full_path, df.shape))
labels = df.columns

# Edible
edible = df["Edible"] == "T"
H = -1 * entropy(edible)

# White
white = df["White"] == 1
is_white = white.sum()
not_white = white.count() - white.sum()
rem_white_hand = -(10/24)*((7/10)*math.log2(7/10) + (3/10)*math.log2(3/10)) - (14/24)*((9/14)*math.log2(9/14) + (5/14)*math.log2(5/14))
rem_white = rem_entropy(white, edible)
ig_white = H - rem_white

# Tall
tall = df["Tall"] == 1
is_tall = tall.sum()
not_tall = len(tall) - is_tall
rem_tall_hand = -(14/24)*((10/14)*math.log2(10/14) + (4/14)*math.log2(4/14)) - (10/24)*((6/10)*math.log2(6/10) + (4/10)*math.log2(4/10))
rem_tall = rem_entropy(tall, edible)
ig_tall = H - rem_tall

# Frilly
frilly = df["Frilly"] == 1
is_frilly = frilly.sum()
not_frilly = len(tall) - is_frilly
rem_frilly_hand = -(8/24)*((3/8)*math.log2(3/8) + (5/8)*math.log2(5/8)) - (16/24)*((13/16)*math.log2(13/16) + (3/16)*math.log2(3/16))
rem_frilly = rem_entropy(frilly, edible)
ig_frilly = H - rem_frilly

print(f'[Entropy] {H:0.4}')
print(f'[Remainder] \nRem White: {rem_white:0.4} \nRem Tall: {rem_tall:0.4} \nRem Frilly: {rem_frilly:0.4}')
print(f'[Information Gain] \nIG White: {ig_white:0.4} \nIG Tall: {ig_tall:0.4} \nIG Frilly: {ig_frilly:0.4}')
   
#%% Decision Tree after is Frilly?
leaf1_df = df[frilly]
leaf1_edible = leaf1_df["Edible"] == "T" 
leaf1_H = -1*entropy(leaf1_edible)
print(f"[Entroypy Frilly True] {leaf1_H:0.4}")

rem_white_leaf1 = rem_entropy(leaf1_df["White"] ==1, leaf1_edible)
rem_tall_leaf1 = rem_entropy(leaf1_df["Tall"] ==1, leaf1_edible)
IG_white_leaf_1 = leaf1_H - rem_white_leaf1
IG_tall_leaf_1 = leaf1_H - rem_tall_leaf1

leaf2_df = df[~frilly]
leaf2_edible = leaf2_df["Edible"] == "T" 
leaf2_H = -1*entropy(leaf2_edible)
print(f"[Entroypy Frilly False] {leaf2_H:0.4}")

rem_white_leaf2 = rem_entropy(leaf2_df["White"] ==1, leaf2_edible)
rem_tall_leaf2 = rem_entropy(leaf2_df["Tall"] ==1, leaf2_edible)
IG_white_leaf_2 = leaf2_H - rem_white_leaf2
IG_tall_leaf_2 = leaf2_H - rem_tall_leaf2

#%% Decision Tree after is Frilly and is White
leaf3_df = leaf1_df[leaf1_df["White"]==1]
leaf3_edible = leaf3_df["Edible"] == "T" 
leaf3_H = -1*entropy(leaf3_edible)


leaf4_df = leaf1_df[~leaf1_df["White"]==1]
leaf4_edible = leaf4_df["Edible"] == "T" 
leaf4_H = -1*entropy(leaf4_edible)


#%% Decision Tree after is Frilly and is Tall
leaf5_df = leaf2_df[leaf2_df["Tall"]==1]
leaf5_edible = leaf5_df["Edible"] == "T" 
leaf5_H = -1*entropy(leaf5_edible)


leaf6_df = leaf2_df[~leaf2_df["Tall"]==1]
leaf6_edible = leaf5_df["Edible"] == "T" 
leaf6_H = -1*entropy(leaf6_edible)
    