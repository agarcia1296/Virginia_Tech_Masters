# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 22:45:55 2022

@author: agarc
"""

import cv2
import numpy as np

#%%
# Part 1
animals = cv2.imread("animals.png")
# Part 2
height = len(animals)
width = len(animals[0])
depth = len(animals[0, 0])
print(f"Animals img size: {animals.shape}")
# Part 3
animals_red = animals[:,:,2]
animals_green = animals[:,:,1]
animals_blue = animals[:,:,0]
print(f"Animals red avg: {np.mean(animals_red)}")
print(f"Animals green avg: {np.mean(animals_green)}")
print(f"Animals blue avg: {np.mean(animals_blue)}")

#%%

# Part 1
stonehenge = cv2.imread("Stonehenge_1024x768.png")
# Part 2
height = len(stonehenge)
width = len(stonehenge[0])
depth = len(stonehenge[0, 0])
print(f"Stonehenge img size: {stonehenge.shape}")
# Part 3
stonehenge_red = stonehenge[:,:,2]
stonehenge_green = stonehenge[:,:,1]
stonehenge_blue = stonehenge[:,:,0]
print(f"Stonehenge red avg: {np.mean(animals_red)}")
print(f"Stonehenge green avg: {np.mean(animals_green)}")
print(f"Stonehenge blue avg: {np.mean(animals_blue)}")

#%% Pixel by Pixel Avg

def avg_pixel_by_pixel(img1, img2):
    # Assuming both images are the same size
    height = len(img1)
    width = len(img1[0])
    new_image = np.zeros((height, width), dtype = "uint8")
    for y_pix in range(height):
        for x_pix in range(width):
            img1_value = img1[y_pix][x_pix]
            img2_value = img2[y_pix][x_pix]
            avg_value = np.mean([img1_value, img2_value])
            new_image[y_pix, x_pix] = avg_value
    return new_image

# Part 4
animals_grayscale = cv2.cvtColor(animals, cv2.COLOR_BGR2GRAY)
stonehenge_grayscale = cv2.cvtColor(stonehenge, cv2.COLOR_BGR2GRAY)

# Part 5
avg_image = avg_pixel_by_pixel(animals_grayscale, stonehenge_grayscale)
# show images
cv2.namedWindow('INPUT', flags=cv2.WINDOW_NORMAL)
cv2.imshow('INPUT',avg_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("part5_img.png", avg_image)

#%% Pixel by Pixel Max

def max_pixel_by_pixel(img1, img2):
    # Assuming both images are the same size
    height = len(img1)
    width = len(img1[0])
    new_image = np.zeros((height, width), dtype = "uint8")
    for y_pix in range(height):
        for x_pix in range(width):
            img1_value = img1[y_pix][x_pix]
            img2_value = img2[y_pix][x_pix]
            max_value = np.max([img1_value, img2_value])
            new_image[y_pix, x_pix] = max_value
    return new_image

# Part 6
max_image = max_pixel_by_pixel(animals_grayscale, stonehenge_grayscale)
# show images
cv2.namedWindow('INPUT', flags=cv2.WINDOW_NORMAL)
cv2.imshow('INPUT',max_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("part6_img.png", max_image)


#%% Pixel by Pixel Difference

def diff_pixel_by_pixel(img1, img2):
    # Assuming both images are the same size
    height = len(img1)
    width = len(img1[0])
    new_image = np.zeros((height, width), dtype = "uint8")
    for y_pix in range(height):
        for x_pix in range(width):
            img1_value = img1[y_pix][x_pix]
            img2_value = img2[y_pix][x_pix]
            diff_value = cv2.absdiff(np.array(img1_value), np.array(img2_value))
            new_image[y_pix, x_pix] = diff_value
    return new_image

# Part 7
diff_image = diff_pixel_by_pixel(animals_grayscale, stonehenge_grayscale)
# show images
cv2.namedWindow('INPUT', flags=cv2.WINDOW_NORMAL)
cv2.imshow('INPUT', diff_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("part7_img.png", diff_image)



