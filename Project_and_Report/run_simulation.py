
import pybullet as p
import pybullet_data
import time
import cv2
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

#------------------------------------------------------------------------------

def capture_image(folder_path, filename):
    # Set up camera parameters
    width, height = 640, 480
    # Camera parameters
    cameraEyePosition = [0, 0, 5]  # Camera position
    cameraTargetPosition = [0, 0, 0]  # Point camera is looking at
    cameraUp = [0, 1, 0]  # Up vector
    
    # Calculate the view matrix
    view_matrix = p.computeViewMatrix(cameraEyePosition, cameraTargetPosition, cameraUp)
    projection_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=float(width)/height, nearVal=0.1, farVal=100)

    # Get the camera image
    _, _, rgb, _, _ = p.getCameraImage(width, height, view_matrix, projection_matrix, shadow=True, renderer=p.ER_BULLET_HARDWARE_OPENGL)

    # Save the image to file
    create_folder_if_not_exists(folder_path)
    image_array = np.reshape(rgb, (height, width, 4))
    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(folder_path,filename), image_bgr)
    
#------------------------------------------------------------------------------

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

#------------------------------------------------------------------------------
        
# Define a callback function to capture mouse events
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param['clicked_point'] = (x, y)

#------------------------------------------------------------------------------

def calculate_final_pixel_location(final_frame_path):
    # Load image
    init_image = cv2.cvtColor(cv2.imread('zero_frame.png'), cv2.COLOR_BGR2GRAY)
    final_image = cv2.cvtColor(cv2.imread(final_frame_path), cv2.COLOR_BGR2GRAY)
    subtracted_image = init_image - final_image
    _, thresholded_image = cv2.threshold(subtracted_image, 25, 255, cv2.THRESH_BINARY)
    
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter out small contours
    min_contour_area = 5  # Minimum contour area to consider as a circle
    circles = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

    # Calculate the centroid of the circle contour
    if circles:
        circle_contour = circles[0]  # Assuming only one circle is detected
        M = cv2.moments(circle_contour)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        print("Center Location (x, y):", cx, cy)
    else:
        print("No circles detected")
    
    return cx, cy, final_image

#------------------------------------------------------------------------------

def calculate_pixel_location(frame_path):
    # Load Zero Frame image
    ittr_folder = os.path.split(frame_path)[0]
    zeroframe_path = os.path.join(ittr_folder, 'zero_frame.jpg')
    init_image = cv2.cvtColor(cv2.imread(zeroframe_path), cv2.COLOR_BGR2GRAY)
    
    # Load frame
    final_image = cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5,5),np.float32)/25
    dst = cv2.filter2D(final_image,-1,kernel)
    
    # Subtract
    subtracted_image = init_image - final_image
    sub = init_image - dst
    _, thresholded_image = cv2.threshold(subtracted_image, 50, 255, cv2.THRESH_BINARY)
    
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter out small contours
    min_contour_area = 5  # Minimum contour area to consider as a circle
    circles = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

    # Calculate the centroid of the circle contour
    if circles:
        circle_contour = circles[0]  # Assuming only one circle is detected
        M = cv2.moments(circle_contour)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        print("Center Location (x, y):", cx, cy)
    else:
        print("No circles detected")
        TypeError('NO CIRCLES!!')
        cx, cy = None, None
    
    return cx, cy 

#------------------------------------------------------------------------------

def main(folder_path_main):
    create_folder_if_not_exists(folder_path_main)
    column_names = ['ittr', 'force_x', 'force_y', 'force_z', 'velocity_x', 'velocity_y', 'velocity_z', 'curr_pos_x', 'curr_pos_y', 'curr_pos_z']
    dataframe = pd.DataFrame(columns = column_names)
    
    for i in tqdm(range(1000)):
        # Current itteration folder path
        ittr_folder_path = os.path.join(folder_path_main,f'ittr{i:03}')
        create_folder_if_not_exists(ittr_folder_path)
        
        p.connect(p.GUI)  # Use p.DIRECT if you don't need a GUI
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Set the path to pybullet_data
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)  # Set gravity in Z direction
        p.setTimeStep(1.0 / 60.0)  # Set the simulation time step
        plane_id = p.loadURDF("plane.urdf")  # Load the ground plane
        
        # Capture Zero Frame
        capture_image(ittr_folder_path, 'zero_frame.jpg')
        
        
        sphere_start_pos = [0, 0, 0.1]
        sphere_id = p.loadURDF("sphere_small.urdf", sphere_start_pos, useFixedBase=False) 
        
        # Apply an initial force to the sphere_small to make it move
        xmin, xmax = -20, 20
        ymin, ymax = -20, 20
        zmin, zmax = 10, 20
        
        initial_force = [np.random.randint(xmin,xmax), np.random.randint(ymin,ymax), np.random.randint(zmin,zmax)]  # Apply a force 
        force_application_point = [0, 0, 0]  # The point where the force is applied
        p.applyExternalForce(sphere_id, -1, forceObj=initial_force, posObj=force_application_point, flags=p.LINK_FRAME)
        
        
        

        # Capture image before pushing
        frame = 0
        term_pos = False
        while not term_pos:
            # Save Image
            capture_image(ittr_folder_path, f"frame_{frame:03}.jpg")
            full_img_path = os.path.join(ittr_folder_path, f"frame_{frame:03}.jpg")
            
            # DataFrame list to be added to df
            df_list = []
            df_list = [i]
            df_list.extend(initial_force)
            
            # Get Velocity of Object
            v, _ = p.getBaseVelocity(sphere_id)
            v_x, v_y, v_z = v
            df_list.extend(v)
            
            # Get the position and orientation of the object
            position, orientation = p.getBasePositionAndOrientation(sphere_id)
            curr_pos_x, curr_pos_y, curr_pos_z = position
            df_list.extend(position)

            '''
            # Get pixel location
            cx, cy = calculate_pixel_location(os.path.join(ittr_folder_path, f'frame_{frame:03}.jpg'))
            
            # Save circle image
            # Draw the circle and its center on the original image
            image = cv2.cvtColor(cv2.imread(full_img_path), cv2.COLOR_BGR2GRAY)
            circle_img = cv2.circle(image, (cx, cy), 10, (0, 0, 255), 2)  # Draw the center in red
            circle_folder_path = os.path.join(ittr_folder_path,'circle_detection')
            circle_img_filename = os.path.join(circle_folder_path, f'circle_detection_frame_{frame}.jpg')
            create_folder_if_not_exists(circle_folder_path)
            cv2.imwrite(circle_img_filename, circle_img)

            df_list.append(cx)
            df_list.append(cy)
            '''
            
            # Save to DF
            series = pd.Series(df_list, index=column_names)
            dataframe = pd.concat([dataframe, pd.DataFrame([df_list], columns = column_names)], ignore_index=True)
            
            
            # Step the simulation forward
            p.stepSimulation()
            time.sleep(1.0 / 60.0) 
            # Check if sphere_id reaches the z=0 plane
            if position[2] <= 0.05 and frame > 10:
                term_pos = True
            # increment frame count by one
            frame += 1 
        
        p.disconnect()
        
        # Save pixel location of final frame
        #final_frame_path = full_img_path
        #cx, cy, circle_image = calculate_final_pixel_location(final_frame_path)
        #df_list.append(cx)
        #df_list.append(cy)
        
        #df_list.append(frame)
        
        
        
        
        #cv2.imwrite(f'images_stop_at_drop_2/circle_detection/circle_detection_{i}.jpg', circle_image)
    # Save final df
    dataframe.to_csv(os.path.join(folder_path_main, 'ittr_df.csv'))
    
        
#------------------------------------------------------------------------------       
        
if __name__ == "__main__":
    import sys
    
    main('training_data')
    

#------------------------------------------------------------------------------
# Archive
             
'''  
# Create the display window
cv2.namedWindow("Circle Detection")
    
# Set up the mouse callback
#mouse_params = {'clicked_point': None}
#cv2.setMouseCallback("Circle Detection", mouse_callback, param=mouse_params)
    
cv2.imshow(f"Circle Detection Sim {i}", final_image)

print('Did the circle detection pass? (y/n)')
# Display the result and wait for user input
while True:
    key = cv2.waitKey(10)
    if key == ord('y'):  # User presses 'y' for yes
        cv2.destroyAllWindows()
        break
    elif key == ord('n'):  # User presses 'n' for no
        cv2.destroyAllWindows()
        return None, None
cv2.destroyAllWindows()
'''