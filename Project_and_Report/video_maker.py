# -*- coding: utf-8 -*-
"""
Purpose: Takes the existing data and turns it into a video
"""
import cv2 as cv
import os
import re


def update_old_folders():
    path = 'images_stop_at_drop'
    dir_list = os.listdir(path)
    
    for ittr_folder in dir_list:
        if ittr_folder == 'circle_detection':
            continue
        _, num = re.split('ittr', ittr_folder)
        num = int(num)
        new_dir_name = os.path.join(path, f'ittr{num:03}')
        os.rename(os.path.join(path, ittr_folder), new_dir_name)
        
        file_list = os.listdir(new_dir_name)
        for file in file_list:
            _, num = re.split('frame_', file)
            num, _ = re.split('.png', num)
            num = int(num)
            new_file_name = os.path.join(new_dir_name, f'frame_{num:03}.png')
            os.rename(os.path.join(path, ittr_folder, file), new_file_name)





# Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'mp4v')
fps = 60
video = cv.VideoWriter('output.mp4', fourcc, fps, (640, 480))

path_list = [r'images_stop_at_drop\ittr000', r'images_stop_at_drop\ittr001', r'images_stop_at_drop\ittr002', r'images_stop_at_drop\ittr003']
circle_list = [r'images_stop_at_drop\circle_detection\circle_detection_0.jpg', r'images_stop_at_drop\circle_detection\circle_detection_1.jpg', 
               r'images_stop_at_drop\circle_detection\circle_detection_2.jpg', r'images_stop_at_drop\circle_detection\circle_detection_3.jpg']
for ittr_folder, circle_img in zip(path_list, circle_list):
    file_list = os.listdir(ittr_folder)
    for frame in file_list:
        full_path = os.path.join(ittr_folder, frame)
        img = cv.imread(full_path)
        video.write(img)
video.release()
            
