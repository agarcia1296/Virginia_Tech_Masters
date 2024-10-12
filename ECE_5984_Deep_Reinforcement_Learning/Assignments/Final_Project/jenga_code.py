# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 12:56:36 2023

@author: agarc
"""

import pybullet as p
import pybullet_data
import time
import pygame

def main():
    # Define game parameters
    tower_height = 18
    block_size = 0.075
    num_blocks_per_layer = 3
    
    # Initialize the physics engine
    p.connect(p.GUI)  # Use the GUI mode for visualization
    p.setGravity(0, 0, -9.81)  # Set gravity
    p.setRealTimeSimulation(0)  # Turn off real-time simulation
    
    # Define block properties
    block_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[block_size/2, block_size/2, block_size/2])
    block_visual = -1  # Use the same shape for collision and visual representation
    block_mass = 0.1
    
    # Create the tower
    tower = []
    for i in range(tower_height):
        for j in range(num_blocks_per_layer):
            x = (j - num_blocks_per_layer/2) * block_size
            y = (i + 0.5) * block_size
            z = block_size/2
            block_id = p.createMultiBody(baseMass=block_mass, baseCollisionShapeIndex=block_shape,
                                         baseVisualShapeIndex=block_visual, basePosition=[x, y, z])
            tower.append(block_id)
    
    # Start the game loop
    while True:
        # Get the user input to select a block
        # TODO: Implement user input
        selected_block_id = None
        while not selected_block_id:
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONUP:
                    # Get the position of the mouse click
                    x, y = event.pos
        
                    # Use PyBullet to raycast from the camera to the scene to find the block under the mouse cursor
                    ray_from = p.getDebugVisualizerCamera()[10]  # Get the camera position
                    ray_to = p.getRayFromPixel(x, y)  # Get the direction of the ray from the mouse cursor
                    ray_hit = p.rayTest(ray_from, ray_to)  # Raycast from the camera to the scene
        
                    # Check if the ray hit a block, and select it if it did
                    for hit in ray_hit:
                        if hit[0] in tower:  # Check if the hit object is a block in the tower
                            selected_block_id = hit[0]  # Select the block
                            break
    
        # Check if the tower is stable
        is_stable = True
        for i in range(len(tower)):
            contact_points = p.getContactPoints(tower[i])
            if len(contact_points) > 0:
                is_stable = False
                break
        if is_stable:
            print("You Win!")
            break
    
        # Simulate physics for one step
        p.stepSimulation()
    
        # Sleep to control the simulation speed
        time.sleep(1/240)
        
    # Clean up
    p.disconnect()


if __name__ == '__main__':
    main()