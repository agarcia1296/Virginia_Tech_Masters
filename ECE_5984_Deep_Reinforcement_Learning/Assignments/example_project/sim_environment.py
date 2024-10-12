# -*- coding: utf-8 -*-
"""
Created on Thu May  4 04:39:28 2023

@author: agarc
"""

import pybullet as p
import sys
import numpy as np
import matplotlib.pyplot as plt
#import imageio
#import pyscreenshot as IG
import os
from math import sin,cos,pi,sqrt
import time


ARM_REACH = .85
ARM_REACH_MIN = .1
ARM_FIRST_HEIGHT = .3

TABLE_HEIGHT = 1.30
POKER_POS_OFFSET = 1.3
GRABBER_POS_OFFSET = 1.3
GRIPPER_LENGTH = .2
BLOCK_HEIGHT = .07
BLOCK_WIDTH = .117
BLOCK_LENGTH = .35


#BLOCK_HEIGHT=.09#These are taken from URDF file
#BLOCK_WIDTH=.15
#BLOCK_LENGTH=.45
#"0.4666666 0.0388888 0.0233333"
POKER_HEIGHT=0.0233333#These are taken from URDF file
POKER_WIDTH=0.0388888
POKER_LENGTH=0.4666666

cam_dist    = 2.5
#cam_yaw    = 60
cam_yaw    = 0
cam_pitch     = 0
cam_pos     = [0,0,2.5]

class sim_environment():
    #I am pre-defining all values here, the values here should not affect anything,
    #But I like it as a reference for what the names of parameters are
    
    #Defaults, can be overwritten in initialization
    outputFilesPath = 'C:/Users/SBWork/Documents/Files/school/SP17/ECE285/Project/'

    pybulletPath = "C:/Users/SBWork/Documents/pythonLibs/bullet3/data/"

    towerWidth = 0
    towerHeight = 0
    towerBlocks = 0
    towerPos = [0,0,0]
    towerOrient = 0
    useGUI = False
    useGripperBot = False
    usePokerBot = False
    buildTower = True
    SIM_SECOND_STEPS = 0
    max_delta = 0
    STEP_SIMS = 3#Number of simulation steps to perform each movement, needs to be >1 for stability
    client = 0
    
    log_data = False
    cur_file_num = 0
    RUN_FILE_NAME = 'exp_'
    data_string = ''
    
    
    pokerbot_initial_pos    = [-POKER_POS_OFFSET,0,TABLE_HEIGHT-.2]
    gripperbot_initial_pos     = [GRABBER_POS_OFFSET,0,TABLE_HEIGHT-.2]
    tower_initial_pos       = [0,0,TABLE_HEIGHT]
    
    tableID    = 0
    pokerbotID = 0
    pokerID    = 0
    gripperbotID = 0
    gripperID = 0
            
    blockList = []
    
    
    pokerBotInitBO = [pokerbot_initial_pos,[0,0,0,1]]
    pokerInitBO = [[-1,0.0,1.47],[0,0,0,1]]#If we are using the poker bot, no need to init position, it is connected to robot
    gripperInitBO = [[1,0.0,1.47],[0,0,0,1]]
    gripperBotInitBO = [gripperbot_initial_pos,[0,0,0,1]]
    towerInitBO = []
    pokerBotRestJoint = [.002,-.08,-.002,2.2,-.0001,.7023,.0023]
    pokerBotDesiredJoint = [0,0,0,0,0,0,0]
    pokerBotResetJoint = pokerBotDesiredJoint
    pokerBotDesiredPos = [0,0,0]
    pokerBotInitOrient = pokerBotRestJoint
    
    poker_bot_constraintID = 0;
    
    gripperBotRestJoint = [0,pi/2,0,-pi/2,0,0,0]
    gripperBotDesiredJoint = [0,0,0,0,0,0,0]
    gripperBotResetJoint = gripperBotDesiredJoint
    gripperBotDesiredPos = [0,0,0]
    gripperBotInitOrient = gripperBotRestJoint
        
    def __init__(self,tW=3,tH=6,useGUI=False,usePokerBot=False,useGripper=False,useGripperBot=False,SIM_SECOND_STEPS=1000,towerOrient=0,delta = .001,buildTower=True,pybulletPath="",outfilePath="",log_data=False,init_poker_pos=[-1,0,2],init_gripper_pos=[1,0,2],log_mode='all',use_slow_motion=False,slow_factor=0):
        self.initialize_environment(tW,tH,useGUI,usePokerBot,useGripper,useGripperBot,SIM_SECOND_STEPS,towerOrient,delta,buildTower,pybulletPath,outfilePath,log_data,init_poker_pos,init_gripper_pos,log_mode,use_slow_motion,slow_factor)


    def initialize_environment(self,tW,tH,useGUI=False,usePokerBot=False,useGripper=False,useGripperBot=False,SIM_SECOND_STEPS=1000,towerOrient=0,delta = .001,buildTower=True,pybulletPath ="",outfilePath="",log_data=False,init_poker_pos=[-1,0,2],init_gripper_pos=[1,0,2],log_mode='all',use_slow_motion=False,slow_factor=0):
        
        GUI_ID = -1;
        DIRECT_ID = -1;
        
        #This function sets up the environment, it currently only
        init_time = time.time();
        if(pybulletPath != ""):
            self.pybulletPath = pybulletPath
        if(outfilePath != ""):
            self.outputFilesPath = outfilePath
            
        self.use_slow_motion = use_slow_motion;
        self.slow_factor = slow_factor;
            
            
        self.tablePath = self.pybulletPath + "table/table_mid.urdf"
        self.kukaPath = self.pybulletPath + "kuka_lwr/kuka3.urdf"
        #self.kukaPath = self.pybulletPath + "kuka_iiwa/model.urdf"
        self.jengaPath = self.pybulletPath + "jenga/jenga_mid3.urdf"
        self.pokerPath = self.pybulletPath + "jenga/poker2.urdf"
        self.gripperPath = self.pybulletPath + "gripper/wsg50_one_motor_gripper_new_free_base.sdf" 
            
            
        #pos block XY
        #pos end eff XYZ
        #pos top 3 blocks Z
        #position on moving block robot will need to touch
        #This function


        #tW width of jenga tower
        #tH height of jenga tower
        #useGUI use the GUI to view the simulation (set to false to speed up processing
        
    
    
        #Set environment variables
        self.towerWidth     = tW
        self.towerHeight    = tH
        self.towerBlocks    = tW*tH
        self.towerPos       = [0,0,TABLE_HEIGHT]
        self.towerOrient    = 0
        self.useGUI         = useGUI
        self.useGripper     = (useGripper or useGripperBot) #If gripperbot enabled, I want the gripper
        self.useGripperBot  = (useGripper or useGripperBot) #For now, gripper will always need the robot
        self.usePokerBot    = usePokerBot
        self.SIM_SECOND_STEPS = SIM_SECOND_STEPS
        self.buildTower     = buildTower
        
        
        
        self.max_delta = delta
        self.log_data = log_data
        self.log_mode = log_mode
        
        self.pokerInitBO[0] = init_poker_pos;
        self.gripperInitBO[0] = init_gripper_pos;
        
        #p.set
    
        #Set sim parameters
        
        
        if(self.useGUI):
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
   
        #test = p.loadURDF(pokerPath,[0,0,5],[0,0,0,1])
        #self.step_sim()
        #print('ID = %d\n'%(test))
        #print('numJoints = %d'%(p.getNumJoints(test)))
        #print('jointInfo = %d'%(p.getJointInfo(test)))
        #print('Base Pos Orient: ')print(p.getBasePositionAndOrientation(test))
        
        p.setTimeStep(1.0/SIM_SECOND_STEPS)
        p.setGravity(0.0,0.0,-9.8)
        #p.resetDebugVisualizerCamera(5,40,0,[-.0376,0.3159,-.0344])
        p.resetDebugVisualizerCamera(cam_dist,cam_pitch,cam_yaw,cam_pos)
        
    
        #Load sim objects
        print('Environment loading objects')
        self.tableID    = p.loadURDF(self.tablePath)
        if(usePokerBot):#If we are not counting the robot, we need the base of the block to be fixed
            self.pokerbotID = p.loadURDF(self.kukaPath,self.pokerbot_initial_pos,useFixedBase=True)
        else:
            self.pokerbotID = 0
            
        if(self.usePokerBot):
            self.pokerID    = p.loadURDF(self.pokerPath,[0,0,5],p.getQuaternionFromEuler([0,0,0]))
        else:
            self.pokerID    = p.loadURDF(self.pokerPath,init_poker_pos,p.getQuaternionFromEuler([0,0,0]),useFixedBase=True)
        
        #self.pokerID    = p.loadSDF(gripperPath)[0]
        
        if(self.useGripper):
            self.gripperInitBO[0] = init_gripper_pos;
            self.gripperInitBO[1] = p.getQuaternionFromEuler([pi/2,0,-pi/2]);
            if(self.useGripperBot):
                self.gripperID = p.loadSDF(self.gripperPath)[0]
            else:
                self.gripperID = p.loadSDF(self.gripperPath)[0]
            p.resetBasePositionAndOrientation(self.gripperID,self.gripperInitBO[0],p.getQuaternionFromEuler([pi/2,0,-pi/2]));
        else:
            self.gripperID = 0
        
        if(self.useGripperBot):
            self.gripperbotID = p.loadURDF(self.kukaPath,self.gripperBotInitBO[0],useFixedBase=True)
        else:
            self.gripperbotID = 0
        
        if(buildTower):
            self.blockList = self.place_tower(self.towerWidth,self.towerHeight,self.towerPos,self.towerOrient)
        else:
            print('Not building tower, this will probably cause issues')
        
        #Attach end-effectors to their robots, set robots to hold orient
        print('Environment setting contstraints')
        if(self.usePokerBot):
            #Poker s
            pokerOrient = p.getQuaternionFromEuler([pi/2,0,0])
            self.poker_bot_constraintID = p.createConstraint(self.pokerbotID,6,self.pokerID,-1,p.JOINT_FIXED,[0,0,0],[0,0,0],[-POKER_LENGTH/2,0,0],childFrameOrientation=p.getQuaternionFromEuler([pi/2,pi,pi/2]))
            p.changeConstraint(self.poker_bot_constraintID,maxForce=100)
            for i in range(0,p.getNumJoints(self.pokerbotID)):
                p.setJointMotorControl2(self.pokerbotID,i,controlMode=p.POSITION_CONTROL,targetPosition=self.pokerBotInitOrient[i],positionGain=1)
            #p.setJointMotorControl2(self.pokerbotID,6,controlMode=p.POSITION_CONTROL,targetPosition=pi/2,positionGain=1)
        
        if(self.useGripperBot):
            gripper_rotation = p.getQuaternionFromEuler([0,0,0]);
            cid = p.createConstraint(self.gripperbotID,6,self.gripperID,-1,p.JOINT_FIXED,[0,0,0],[0,0.005,0.2],[0,.01,0.2],childFrameOrientation=gripper_rotation)
            p.changeConstraint(cid,maxForce=5000)
            for i in range(0,p.getNumJoints(self.gripperbotID)):
                p.setJointMotorControl2(self.gripperbotID,i,controlMode=p.POSITION_CONTROL,targetPosition=self.gripperBotInitOrient[i],positionGain=1)
                
            self.close_gripper()
                
        
        #Run a very short period of time, just so everything settles into position
        #p.setRealTimeSimulation(enableRealTimeSimulation = 1)
        for i in range(0,100):
            self.step_sim()
                
                
        print('Getting initial positions of all objects')
        
        if(buildTower):        
            self.towerInitBO = []
            for i in range(0,self.towerBlocks):
                self.towerInitBO.append(p.getBasePositionAndOrientation(self.blockList[i]))
        else:
            print('Not building tower, this will probably cause issues')
              
        self.begin_log() #Logging is always performed, it is only saved to file if log_data is used
        
        self.set_poker_reset_position(init_poker_pos)
        
        if(self.useGripper):
            self.set_gripper_reset_position(init_gripper_pos)
        
        
        print('gripper reset pos');
        self.reset_simulation();
        total_time = time.time() - init_time;
        
        
        
        
        print('Environment setup complete, took: %s seconds'%(total_time))
    

    def log_step(self,val):
        #This stores the action into the list
        self.data_string += val
    def flush_log(self,filePath = ''):
        #Use this function to write log string to file, 
        if(filePath==''):
            filePath = self.outputFilesPath + self.RUN_FILE_NAME + str(self.cur_file_num) + '.bin'
        file = open(filePath,'wb')
        file.write(self.data_string)
        self.clear_log()
        file.close() 
    def clear_log(self):
        #Use this function to clear log string, use if you are handling file writes externally
        self.data_string = ''
    def begin_log(self):
        #This should be called during reset, you should not need to call it yourself
        self.clear_log()
        self.cur_file_num+=1
    def get_log_string(self):
        return self.data_string
        
        
    def recreate_run(self,filePath='',string=''):
        #log_setting = self.log_data
        #self.log_data = False
        
        if(string=='' and filePath!=''):
            file = open(filePath,'r')
            string = file.read()
            file.close()
        
        self.reset_simulation()
        for c in string:
            if(c == 'F'):
                self.move_poker_px(False)
            elif(c == 'B'):
                self.move_poker_nx(False)
            elif(c == 'L'):
                self.move_poker_py(False)
            elif(c == 'R'):
                self.move_poker_ny(False)
            elif(c == 'U'):
                self.move_poker_pz(False)
            elif(c == 'D'):
                self.move_poker_nz(False)
            elif(c == 'S'):
                self.move_poker_stationary(False)
            else:
                print('Error reading run file, unrecognized character %s'%(c))
            
                
        #self.log_data = log_setting
        
    def reset_simulation(self,ignore_log = False):       
        
        
        self.reset_poker_position()
        #raw_input();
        for i in range(0,5):
            self.step_sim()
        
        
        
        if(self.useGripperBot):
            #Gripper gripper will always be attached to gripper bot, no need to init BO
            self.reset_gripper_position()
            
        if(self.buildTower):    
            for i in range(0,self.towerBlocks):
                p.resetBasePositionAndOrientation(self.blockList[i],self.towerInitBO[i][0],self.towerInitBO[i][1])
        

        #Run a very short period of time, just so everything settles into position
        #p.setRealTimeSimulation(enableRealTimeSimulation = 1)
        for i in range(0,100):
            self.step_sim()
            
        
        #log_data now only has an effect on if data is stored to file, it is always collected        
        if(self.log_data and not ignore_log):    
            self.flush_log()
            
        self.begin_log()
            
    #BEGIN RESET STUFF CODE
    
    def set_gripper_reset_position(self,position=[9454]):
        if(position[0] == 9454):#Use current position
            position = self.getBasePositionAndOrientation(self.gripperID)[0]
            
        self.gripperInitBO[0] = position
        
        if(self.useGripperBot):
            self.set_gripper_position(position,1000000)
            for i in range(0,300):
                    self.step_sim()
            self.set_gripperbot_reset_joints()
    def set_gripperbot_reset_joints(self,joints=[10]):
        if(joints[0]== 10): #Use current robot positions
            for i in range(0,7):
                self.gripperBotResetJoint[i] = p.getJointState(self.gripperbotID,i)[0]
        else:               #Use given robot positions
            self.gripperBotResetJoint = joints
    def set_poker_reset_position(self,position=[9454]):
        
        if(position[0] == 9454):#Use current position
            position = self.getBasePositionAndOrientation(self.pokerID)[0]
            
        self.pokerInitBO[0] = position
        
        if(self.usePokerBot):
            self.set_poker_position(position,1000000)
            for i in range(0,300):
                    self.step_sim()
            #OR = self.pokerInitBO[1]
            #jd = [.00001,.00001,.00001,.00001,.00001,.00001,.00001]
            #joints = p.calculateInverseKinematics(self.pokerbotID,6,targetPosition=position,targetOrientation=OR,jointDamping=jd,restPoses=self.pokerBotRestJoint)
            self.set_pokerbot_reset_joints()
            
    def set_pokerbot_reset_joints(self,joints=[10]):
        
        if(joints[0]== 10): #Use current robot positions
            for i in range(0,7):
                self.pokerBotResetJoint[i] = p.getJointState(self.pokerbotID,i)[0]
        else:               #Use given robot positions
            self.pokerBotResetJoint = joints
            
    def reset_poker_position(self,realTime=False):
        self.pokerBotDesiredJoint = self.pokerBotResetJoint
        self.pokerBotDesiredPos = np.subtract(self.pokerInitBO[0],[POKER_LENGTH,0,0]);
        if(self.usePokerBot):
            p.changeConstraint(self.poker_bot_constraintID,maxForce = 0)
            for i in range(0,10):
                self.step_sim()
            p.changeConstraint(self.poker_bot_constraintID,maxForce = 10000)
            for i in range(0,300):
                self.step_sim()
            if(realTime):
                for i in range(0,250):
                    self.step_sim()
            else:
                for i in range(0,7):
                    p.resetJointState(self.pokerbotID,i,self.pokerBotResetJoint[i])
                    
        else: #Pokerbot not used, just change poker position
        
            offset = np.subtract(self.get_poker_center_position(),self.get_poker_position());
            pos = np.add(offset,self.pokerInitBO[0]);
            p.resetBasePositionAndOrientation(self.pokerID,pos,self.pokerInitBO[1])
 

    def reset_gripper_position(self,realTime=False):
        self.gripperBotDesiredJoint = self.gripperBotResetJoint
        self.gripperBotDesiredPos = self.gripperInitBO[0]
        if(self.useGripperBot):
            
            if(realTime):
                for i in range(0,250):
                    self.step_sim()
            else:
                
                for i in range(0,7):
                    p.resetJointState(self.gripperbotID,i,self.gripperBotResetJoint[i])
        else: #Pokerbot not used, just change poker position
        
            offset = np.subtract(self.get_gripper_center_position(),self.get_gripper_position());
            pos = np.add(offset,self.gripperInitBO[0]);
            p.resetBasePositionAndOrientation(self.gripperID,pos,self.gripperInitBO[1]) 
    #BEGIN CONTROL POKER CODE---------------------------------------------------
    
    
    
    
    def set_poker_position(self,position,force=-1):
        #This sets the position based on block end, not block center
        #if(True):
        if(not self.usePokerBot):
            endPos,OR = self.get_poker_position_and_orientation()
            cenPos = self.get_poker_center_position()
            diff = np.subtract(cenPos,endPos)
            newPos = np.add(position,diff)
            OR = self.pokerInitBO[1]
            
            
            
            if(len(newPos)==1):
                newPos = newPos[0]#This is here because sometimes nump outputs array within array
            p.resetBasePositionAndOrientation(self.pokerID,newPos,OR)
            
            
        else:
                
            endPos = self.get_poker_back_position_and_orientation(use_actual_OR=False)[0]
            cenPos = self.get_poker_position(False)
            #cenPos = self.get_poker_center_position()
            diff = np.subtract(endPos,cenPos)
            #print('DIFF')
            #print(diff)
            
            newPos = np.add(position,diff)
            self.pokerBotDesiredPos = newPos
            #newPos = position
            jd = [.00001,.00001,.00001,.00001,.00001,.00001,.00001]
            #jd = [100,100,100,100,100,100,100]
            OR = p.getQuaternionFromEuler([0,pi/2,0])
            
            
            
            for j in range(0,15):
                jointPos = p.calculateInverseKinematics(self.pokerbotID,6,targetPosition=newPos,targetOrientation=OR,jointDamping=jd,restPoses=self.pokerBotRestJoint)
            
                self.pokerBotDesiredJoint = jointPos
                #print(jointPos)
                self.set_pokerBot_position(jointPos,force)
                #Do inverse kinematics on robot
                for i in range(0,10):
                    self.step_sim()
            #print('NewMove')
            #print(jointPos)
            #print('Back Values')
            #print(newPos)
            #print(self.get_poker_back_position_and_orientation()[0])
            #print(np.subtract(self.get_poker_back_position_and_orientation()[0],newPos))
            
            #print('Front Values')
            #print(position)
            #print(self.get_poker_position_and_orientation()[0])
            #print(np.subtract(self.get_poker_position_and_orientation()[0],position))
            #print('End \n')
            #print(self.get_poker_back_position_and_orientation()[0])
            #print(p.getEulerFromQuaternion(self.get_poker_back_position_and_orientation()[1]))
            #print(self.get_poker_position_and_orientation()[0])
            #print(p.getEulerFromQuaternion(self.get_poker_position_and_orientation()[1]))
            
    def set_gripper_position(self,position,force=-1):
        #This sets the position based on block end, not block center
        #if(True):
        if(not self.useGripperBot):
            endPos,OR = self.get_gripper_position_and_orientation()
            cenPos = self.get_gripper_center_position()
            diff = np.subtract(cenPos,endPos)
            newPos = np.add(position,diff)
            OR = self.gripperInitBO[1]
            
            
            
            if(len(newPos)==1):
                newPos = newPos[0]#This is here because sometimes nump outputs array within array
            p.resetBasePositionAndOrientation(self.gripperID,newPos,OR)
            
            
        else:
            endPos = self.get_gripper_back_position_and_orientation(use_actual_OR=False)[0]
            cenPos = self.get_gripper_position(use_actual_OR=False)
            #cenPos = self.get_poker_center_position()
            diff = np.subtract(endPos,cenPos)
            #print('DIFF')
            #print(diff)
            
            
            newPos = np.add(position,diff)
            self.gripperBotDesiredPos = newPos
            #newPos = position
            jd = [.00001,.00001,.00001,.00001,.00001,.00001,.00001]
            #jd = [100,100,100,100,100,100,100]
            OR = p.getQuaternionFromEuler([pi/2,0,-pi/2])
            
            for j in range(0,15):
                jointPos = p.calculateInverseKinematics(self.gripperbotID,6,targetPosition=newPos,targetOrientation=OR,jointDamping=jd,restPoses=self.gripperBotResetJoint)
            
                self.gripperBotDesiredJoint = jointPos
                #print(jointPos)
                self.set_gripperBot_position(jointPos,force)
                #Do inverse kinematics on robot
                for i in range(0,10):
                    self.step_sim()    
    def set_poker_orientation(self,orientation):
        print('set poker orientation not yet implemented')
        #return [0,0,0,1]
        
    def set_block_position(self,ID,position):
        OR = self.towerInitBO[ID][1]
        p.resetBasePositionAndOrientation(self.blockList[ID],position,OR)
    def set_pokerBot_position(self,jointPos,force=-1):
        if force < 0:
            force = 1000
        for i in range(0,7):
                p.setJointMotorControl2(self.pokerbotID,i,controlMode=p.POSITION_CONTROL,targetPosition=jointPos[i],force=force,positionGain=.05,velocityGain=1)
    def set_gripperBot_position(self,jointPos,force=-1):
        if force < 0:
            force = 1000
        for i in range(0,7):
                p.setJointMotorControl2(self.gripperbotID,i,controlMode=p.POSITION_CONTROL,targetPosition=jointPos[i],force=force,positionGain=.05,velocityGain=1)
    def move_poker(self,offset):
        
        #if(True):
        if(not self.usePokerBot):
            pos,OR = p.getBasePositionAndOrientation(self.pokerID)
            p.resetBasePositionAndOrientation(self.pokerID,np.add(pos,offset),OR)
            #for i in range(0,self.STEP_SIMS):
            #    self.step_sim()
                
        else:
            #pos,OR = self.get_poker_back_position_and_orientation(False)
            #new_pos = np.add(pos,offset)
            new_pos = np.add(self.pokerBotDesiredPos,offset)
            
            if(not self.check_range(new_pos)):
                new_pos = self.pokerBotDesiredPos
            
            self.pokerBotDesiredPos = new_pos
            jd = [.00001,.00001,.00001,.00001,.00001,.00001,.00001]
            #jd = [100,100,100,100,100,100,100]
            OR = p.getQuaternionFromEuler([0,pi/2,0])
            #print(new_pos)
            jointPos = p.calculateInverseKinematics(self.pokerbotID,6,targetPosition=new_pos,targetOrientation=OR,jointDamping=jd,restPoses=self.pokerBotRestJoint)
            self.pokerBotDesiredJoint = jointPos
            #print(jointPos)
            self.set_pokerBot_position(jointPos)
            
        for i in range(0,self.STEP_SIMS):
            self.step_sim()
            
    def move_gripper(self,offset):
        
        #if(True):
        if(not self.useGripperBot):
            pos,OR = p.getBasePositionAndOrientation(self.gripperID)
            p.resetBasePositionAndOrientation(self.gripperID,np.add(pos,offset),OR)
            #for i in range(0,self.STEP_SIMS):
            #    self.step_sim()
                
        else:
            #pos,OR = self.get_poker_back_position_and_orientation(False)
            #new_pos = np.add(pos,offset)
            new_pos = np.add(self.gripperBotDesiredPos,offset)
            if(not self.check_range_gripper(new_pos)):
                new_pos = self.gripperBotDesiredPos
            
            self.gripperBotDesiredPos = new_pos
            #print(new_pos)
            jd = [.00001,.00001,.00001,.00001,.00001,.00001,.00001]
            #jd = [100,100,100,100,100,100,100]
            OR = self.gripperInitBO[1]
            #print(new_pos)
            jointPos = p.calculateInverseKinematics(self.gripperbotID,6,targetPosition=new_pos,targetOrientation=OR,jointDamping=jd,restPoses=self.gripperBotRestJoint)
            self.gripperBotDesiredJoint = jointPos
            #print(jointPos)
            self.set_gripperBot_position(jointPos)
            
        for i in range(0,self.STEP_SIMS):
            self.step_sim()    
            
    def move_poker_px(self,log=True):
        if(log):
            self.log_step('F')
        self.move_poker([self.max_delta,0,0])
        
    def move_poker_nx(self,log=True):
        if(log): 
            self.log_step('B')
        self.move_poker([-self.max_delta,0,0])
        
    def move_poker_py(self,log=True):
        if(log): 
            self.log_step('L')
        self.move_poker([0,self.max_delta,0])
        
    def move_poker_ny(self,log=True):
        if(log): 
            self.log_step('R')
        self.move_poker([0,-self.max_delta,0])
    
    def move_poker_pz(self,log=True):
        if(log): 
            self.log_step('U')
        self.move_poker([0,0,self.max_delta])
        
    def move_poker_nz(self,log=True):
        if(log): 
            self.log_step('D')
        self.move_poker([0,0,-self.max_delta])
        
    def move_poker_stationary(self,log=True):
        if(log): 
            self.log_step('S')
        self.move_poker([0,0,0])
        
        
    def move_gripper_px(self,log=True):
        if(log):
            self.log_step('F')
        self.move_gripper([self.max_delta,0,0])
        
    def move_gripper_nx(self,log=True):
        if(log): 
            self.log_step('B')
        self.move_gripper([-self.max_delta,0,0])
        
    def move_gripper_py(self,log=True):
        if(log): 
            self.log_step('L')
        self.move_gripper([0,self.max_delta,0])
        
    def move_gripper_ny(self,log=True):
        if(log): 
            self.log_step('R')
        self.move_gripper([0,-self.max_delta,0])
    
    def move_gripper_pz(self,log=True):
        if(log): 
            self.log_step('U')
        self.move_gripper([0,0,self.max_delta])
        
    def move_gripper_nz(self,log=True):
        if(log): 
            self.log_step('D')
        self.move_gripper([0,0,-self.max_delta])
        
    def move_gripper_stationary(self,log=True):
        if(log): 
            self.log_step('S')
        self.move_gripper([0,0,0])
    


    def offset_block(self,ID,amount):
        cenPos = self.get_block_center_position(ID);
        endPos = self.get_block_back_position(ID);
        
        diff = np.subtract(endPos,cenPos)
        diff = diff / np.linalg.norm(diff)
        
        offset_v = diff * amount;
        newPos = np.add(cenPos,offset_v)
        
        self.set_block_position(ID,newPos)
    
    #END CONTROL POKER CODE---------------------------------------------------
    def step_sim(self):
        if(self.usePokerBot):
            self.set_pokerBot_position(self.pokerBotDesiredJoint)
        if(self.use_slow_motion):
            time.sleep(.001*self.slow_factor);    
        p.stepSimulation()
    
    def check_range(self,position):
        
        #Desired Offset
        init_pos = np.add(self.pokerbot_initial_pos,[0,0,ARM_FIRST_HEIGHT])
        d_o= np.subtract(position,init_pos)
        
        distance = d_o[0]*d_o[0] + d_o[1]*d_o[1] + d_o[2]*d_o[2]
        distance = sqrt(distance)
        
        #print("Need to determine bounds that robot can reliably use")
        if(distance > ARM_REACH):
            return False
        
        if(distance < ARM_REACH_MIN):
            return False
        
        return True
        
    def check_range_gripper(self,position):
        
        #Desired Offset
        init_pos = np.add(self.gripperbot_initial_pos,[0,0,ARM_FIRST_HEIGHT])
        d_o= np.subtract(position,init_pos)
        distance = d_o[0]*d_o[0] + d_o[1]*d_o[1] + d_o[2]*d_o[2]
        distance = sqrt(distance)
        
        #print("Need to determine bounds that robot can reliably use")
        if(distance > ARM_REACH):
            return False
        
        if(distance < ARM_REACH_MIN):
            return False
        
        return True
    #BEGIN GET POS/ORIENT CODE-------------------------------------------------------
    
    #Code to get positions / orientations of various objects from the simulation
    #The next 3 functions get the position of the tip of the end effector
    def get_poker_position(self,use_actual_OR=True):
        return self.get_poker_position_and_orientation(use_actual_OR)[0]
    #Orientation is same for tip and center    
    def get_poker_orientation(self):
        return self.get_poker_position_and_orientation()[1]
    def get_poker_position_and_orientation(self,use_actual_OR=True):
        pos,OR = p.getBasePositionAndOrientation(self.pokerID)
        if(not use_actual_OR):
            OR = self.pokerInitBO[1]
        #This additional math is to get tip of end effector, not center
        rot = np.reshape(p.getMatrixFromQuaternion(OR),[3,3])
        block_offset_t = np.reshape([POKER_LENGTH/2,0,0],[1,3])
        block_offset = np.matmul(block_offset_t,rot)
        pos_offset = pos + block_offset
        return pos_offset[0],OR
        
    def get_poker_center_position(self):
        return p.getBasePositionAndOrientation(self.pokerID)[0]  
    def get_poker_center_position_and_orientation(self):
        return p.getBasePositionAndOrientation(self.pokerID)
    def get_poker_back_position(self,use_actual_OR=True):
        return self.get_poker_back_position_and_orientation(use_actual_OR)[0]
        
    def get_poker_back_position_and_orientation(self,use_actual_OR=True):
        pos,OR = p.getBasePositionAndOrientation(self.pokerID)
        op_offset = np.subtract(pos,self.get_poker_position(use_actual_OR))
        back_pos = np.add(pos,op_offset)
        return back_pos,OR
    
    #These functions are for getting position / orientation of particular tower blocks    
    def get_block_position(self,ID):
        return self.get_block_position_and_orientation(ID)[0]
    def get_block_orientation(self,ID):
        return self.get_block_position_and_orientation(ID)[1]
    def get_block_position_and_orientation(self,ID):
        pos_OR = p.getBasePositionAndOrientation(self.blockList[ID])
        OR = pos_OR[1]
        pos = pos_OR[0]
        
        #This additional math is to get tip of end effector, not center
        rot = np.reshape(p.getMatrixFromQuaternion(OR),[3,3])
        block_offset_t = np.reshape([BLOCK_LENGTH/2,0,0],[1,3])
        #TODO: MATRIX MULTIPLICATION
        block_offset = np.matmul(block_offset_t,rot)
        pos_offset = pos - block_offset#Minus in this case because we want the back of the block, not the front
        return pos_offset[0],OR
          
    def get_block_center_position(self,ID):
        return p.getBasePositionAndOrientation(self.blockList[ID])[0]
    def get_block_center_position_and_orientation(self,ID):
        return p.getBasePositionAndOrientation(self.blockList[ID])
    
    def get_block_back_position(self,ID):
        return self.get_block_back_position_and_orientation(ID)[0]
        
    def get_block_back_position_and_orientation(self,ID):
        pos,OR = self.get_block_center_position_and_orientation(ID)
        op_offset = np.subtract(pos,self.get_block_position(ID))
        back_pos = np.add(pos,op_offset)
        return back_pos,OR

    
    def get_gripper_position(self,use_actual_OR=True):
        return self.get_gripper_position_and_orientation(use_actual_OR)[0]
    #Orientation is same for tip and center    
    def get_gripper_orientation(self,use_actual_OR=True):
        return self.get_gripper_position_and_orientation(use_actual_OR)[1]
    def get_gripper_position_and_orientation(self,use_actual_OR=True):
        pos,OR = p.getBasePositionAndOrientation(self.gripperID)
        if(not use_actual_OR):
            OR = self.gripperInitBO[1]
        #This additional math is to get tip of end effector, not center
        rot = np.reshape(p.getMatrixFromQuaternion(OR),[3,3])
        block_offset_t = np.reshape([0,GRIPPER_LENGTH/2,0],[1,3])
        block_offset = np.matmul(block_offset_t,rot)
        pos_offset = pos + block_offset
        return pos_offset[0],OR
        
        
        
    def get_gripper_back_position(self,use_actual_OR=True):
        return self.get_gripper_back_position_and_orientation(use_actual_OR)[0]
        
    def get_gripper_back_position_and_orientation(self,use_actual_OR=True):
        pos,OR = p.getBasePositionAndOrientation(self.gripperID)
        op_offset = np.subtract(pos,self.get_gripper_position(use_actual_OR))
        back_pos = np.add(pos,op_offset)
        
        return back_pos,OR
    

    #END GET POS/ORIENT CODE-------------------------------------------------------

    #BEGIN OTHER PARAMETERS CODE---------------------------------------------------
    def set_movement_delta(self,delta):
        self.max_delta = delta
        
    def get_movement_delta(self):
        return self.max_delta
    
    def get_num_blocks(self):
        return self.towerBlocks
        
    def get_top_blocks_IDS(self):
        num = self.towerBlocks
        return [num-3,num-2,num-1]
        
    def get_good_push_block(self):
        #This will need to be a bit more sophisticated, but it will do for now
        center_off = (self.towerWidth-1) / 2
        row = 3 #Odd if first row faces bot, even otherwise
        if(self.towerHeight <= row):
            row = self.towerHeight
        return int(center_off + (row-1)*self.towerWidth)
    #END OTHER PARAMETERS CODE---------------------------------------------------
    
    def place_tower(self,tW,tH,basePos,baseOrient=0):
        #blockPath is string path to block object
        #tW is width of tower (usually 3, must be odd)
        #tH is height of tower
        #basePos is position of center bottom block
        cosb = cos(baseOrient)
        sinb = sin(baseOrient)
        blockList = []
        BO = (tW-1)/2
        curHeight = basePos[2]
        for i in range(0,tH):
            curHeight = basePos[2]+BLOCK_HEIGHT*i
            curPos = [basePos[0],basePos[1],curHeight]
            for j in range(0,tW):
                #print('(i,j) = (%d,%d)'%(i,j))
                if(i%2 ==0):
                    orientation = p.getQuaternionFromEuler([0,0,0+baseOrient])
                    offset = (j-BO) * BLOCK_WIDTH
                    curPos[0] = basePos[0]+offset*sinb
                    curPos[1] = basePos[1]+offset*cosb
                else:
                    orientation = p.getQuaternionFromEuler([0,0,(pi/2)+baseOrient])
                    offset = (j-BO) * BLOCK_WIDTH
                    curPos[0] = basePos[0]+offset*cosb
                    curPos[1] = basePos[1]+offset*sinb
                #print('Loading block at [%f,%f,%f]'%(curPos[0],curPos[1],curPos[2]))
                blockList.append(p.loadURDF(self.jengaPath,curPos,orientation))
        #print(p.getBodyInfo(blockList[0]))
        return blockList
        
    
    
    def move_kuka_proof_of_concept(R_ID,linkID):
        jointLowerLIMITID = 8
        jointUpperLIMITID = 9
        mode = p.POSITION_CONTROL
        velocity = 0
        maxForce = .0000001
        lower_lim = p.getJointInfo(R_ID,linkID)[8]
        upper_lim = p.getJointInfo(R_ID,linkID)[9]
        steadyState = p.getJointState(R_ID,linkID)[0]
        print('Positions: [%f,%f,%f]'%(lower_lim,upper_lim,steadyState))
        p.setJointMotorControl2(R_ID,linkID,controlMode=mode,targetPosition=lower_lim,force =200)
        for i in range(1,SIM_SECOND):
            moveSim()
            
        p.setJointMotorControl2(R_ID,linkID,controlMode=mode,targetPosition=upper_lim,force =200)
        for i in range(1,SIM_SECOND):
            moveSim()
            
        p.setJointMotorControl2(R_ID,linkID,controlMode=mode,targetPosition=steadyState,force =1000)
        for i in range(1,SIM_SECOND):
            moveSim()
    def close_gripper(self):
        mode = p.POSITION_CONTROL
        GRIPPER_CLOSED=[0.000000,-0.011130,-0.206421,0.205143,0.05,0.000000,0.05,0.000000]
        for i in range(0,8):
             p.setJointMotorControl2(self.gripperID,i,controlMode=mode,targetPosition=GRIPPER_CLOSED[i],force=200)
    
    
    def open_gripper(self):
        mode = p.POSITION_CONTROL
        GRIPPER_OPEN=[0.000000,-0.011130,0.206421,0.205143,-0.01,0.000000,-0.01,0.000000]
        for i in range(0,8):
             p.setJointMotorControl2(self.gripperID,i,controlMode=mode,targetPosition=GRIPPER_OPEN[i])

    def run_jenga_proof_of_concept(self):
        pos = [-.7,0.0,1.47]
        orient = p.getQuaternionFromEuler([0,0,0])
        print('Running Jenga Proof of Concept')
        start_time = time.time()
        env.set_poker_position([-1,0.0,1.47])
        env.set_poker_orientation([0,0,0,1])
        for i in range(0,1000):
            env.move_poker_px()
        elapsed_time = time.time() - start_time
        print('Proof of Concept complete, elapsed time = %ds'%(elapsed_time))