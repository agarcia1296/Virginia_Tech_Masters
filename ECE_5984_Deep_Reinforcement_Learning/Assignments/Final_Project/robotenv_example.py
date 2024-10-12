# -*- coding: utf-8 -*-
"""
Created on Thu May  4 01:54:26 2023

@author: agarc
"""
from RobotArmEnv import RobotArmEnv
import numpy as np

#Example Implementation

env = RobotArmEnv()
rgb_image = env.render(mode = 'human')
env.reset()
env.define_goal(2, np.array([0,1,0]))


#%%
observation, reward, done, info = env.step(env.action_space.low)

#%%
observation, reward, done, info = env.step(env.action_space.high)

#%%
env.close()