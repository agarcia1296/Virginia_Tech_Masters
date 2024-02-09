
import pandas as pd
import numpy as np
import os
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split



#------------------------------------------------------------------------------

def position_to_pixel(position, image_width=640, image_height=480):
    """
    Convert position coordinates to pixel locations.
    
    Args:
    - position: Tuple (x, y) representing the position coordinates.
    - image_width: Width of the image in pixels.
    - image_height: Height of the image in pixels.
    
    Returns:
    - Tuple (pixel_x, pixel_y) representing the pixel locations.
    """
    # Calculate the pixel location
    pixel_x = int((position[0] + 4) * (image_width / 8))
    pixel_y = int((3 - position[1]) * (image_height / 6))
    
    return (pixel_x, pixel_y)


def make_prediction_video(file_name:str, image_path_list:list, actual_final_pos, predicted_final_pos):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 5
    video = cv2.VideoWriter(file_name, fourcc, fps, (640, 480))
    radius = 10  # Radius of the circle
    actual_color = (0, 255, 0)  # green
    pred_color = (0, 0, 255)  # green
    thickness = 2  # Thickness of the circle outline (-1 fills the circle)
    for img_path, actual, predicted in zip(image_path_list, actual_final_pos, predicted_final_pos):
        # Get actual and predicted pixel locations
        actual_x, actual_y, _ = actual
        actual_r, actual_c = position_to_pixel((actual_x, actual_y))
        pred_x, pred_y, _ = predicted
        pred_r, pred_c = position_to_pixel((pred_x, pred_y))

        # Draw circle on image
        image = cv2.imread(img_path)
        cv2.circle(image, (actual_r, actual_c), radius, actual_color, thickness)
        cv2.circle(image, (pred_r, pred_c), radius, pred_color, thickness)
        video.write(image)
    video.release()


#------------------------------------------------------------------------------

# Now process the training and testing videos separately
def process_videos(img_file_paths:list, metadata:list):
    img_data = []
    meta_data = []
    for img_path, meta in tqdm(zip(img_file_paths,metadata), desc = "Processing Images"):
        # Load and preprocess video frame
        video_frame = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
        #frame_resized = tf.image.resize(video_frame, (256,256))
        #video_frame_tensor = torch.tensor(video_frame, dtype=torch.float32)  # Convert frame to tensor
        # Convert metadata to tensor
        #meta_tensor = torch.tensor(meta.astype(float), dtype = torch.float32)
   
        img_data.append(video_frame) 
        meta_data.append(meta)
    return np.array(img_data), np.array(meta_data)

#------------------------------------------------------------------------------

def get_test_train_data():
    # Import Meta Data
    meta_df = pd.read_csv('training_data/FINAL_DF_WITH_PREVIOUS_DATA.csv', index_col=0)
    
    # Drop ittr
    meta_df = meta_df.drop(columns = 'ittr')
    
    # Split into Test and Train data
    target_columns = ['final_pos_x', 'final_pos_y', 'final_pos_z']
    X = meta_df.drop(columns = target_columns).to_numpy()
    num_metadata_features = X.shape[1]
    y = meta_df.drop(columns=meta_df.columns.difference(target_columns)).to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # TRAIN DATA
    # Load video frames and process them
    img_file_paths_train = X_train[:, -1]
    metadata_train =  X_train[:,:-1]
    #img_data_train, meta_data_train = process_videos(img_file_paths_train, metadata_train) 
    #img_data_tensor_train = torch.tensor(img_data_tensor_train, dtype=torch.float32)
    #meta_data_tensor_train = torch.tensor(meta_data_tensor_train.astype(float), dtype=torch.float32)
    
    # TEST DATA
    img_file_paths_test = X_test[:, -1]
    metadata_test =  X_test[:,:-1]
    #img_data_test, meta_data_test = process_videos(img_file_paths_test, metadata_test) 
    #img_data_tensor_test = torch.tensor(img_data_tensor_test, dtype=torch.float32)
    #meta_data_tensor_test = torch.tensor(meta_data_tensor_test.astype(float), dtype=torch.float32)
    
    # TARGET DATA
    # Now y_train contains the target values for your training data
    #targets_train = torch.tensor(y_train, dtype=torch.float32)
    #targets_test = torch.tensor(y_test, dtype=torch.float32)
    
    return X, y, metadata_train, metadata_test, y_train, y_test
    
    #--------------------------------------------------------------------------

def run_linear_regression_model(make_video = False):
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    
    X, y, metadata_train, metadata_test, y_train, y_test = get_test_train_data()
    
    linreg_model = LinearRegression().fit(metadata_train, y_train)
    print(f'R2: {linreg_model.score(metadata_test, y_test)}')
    print(f'MSE: {mean_squared_error(y_test, linreg_model.predict(metadata_test))}')
    
    linreg_model.predict(metadata_test)
   
    if make_video:
        ittr_0_input_data = X[0:32,:-1]
        ittr_0_images = X[0:32,-1]
        ittr_0_target_data = y[0:32]
        model_predictions = linreg_model.predict(ittr_0_input_data)
        make_prediction_video('ittr_0_prediction.mp4', ittr_0_images, ittr_0_target_data, model_predictions)
        
        
        ittr_1_input_data = X[32:64,:-1]
        ittr_1_images = X[32:64,-1]
        ittr_1_target_data = y[32:64]
        model_predictions = linreg_model.predict(ittr_1_input_data)
        make_prediction_video('ittr_1_prediction.mp4', ittr_1_images, ittr_1_target_data, model_predictions)
        
       
        ittr_2_input_data = X[64:88,:-1]
        ittr_2_images = X[64:88,-1]
        ittr_2_target_data = y[64:88]
        model_predictions = linreg_model.predict(ittr_2_input_data)
        make_prediction_video('ittr_2_prediction.mp4', ittr_2_images, ittr_2_target_data, model_predictions)
       
        ittr_3_input_data = X[88:122,:-1]
        ittr_3_images = X[88:122,-1]
        ittr_3_target_data = y[88:122]
        model_predictions = linreg_model.predict(ittr_3_input_data)
        make_prediction_video('ittr_3_prediction.mp4', ittr_3_images, ittr_3_target_data, model_predictions)
   
    #--------------------------------------------------------------------------
    
def run_MLPRegressor_model(train_model = False, make_video = False):
    from sklearn.neural_network import MLPRegressor
    from sklearn.metrics import mean_squared_error
    from itertools import product
    from datetime import datetime
    
    X, y, X_train, X_test, y_train, y_test = get_test_train_data()
    
    if train_model:
        t1 = datetime.now()
        resultsDF = pd.DataFrame(data=None, columns=['Hidden Layer', 'Activation', 'Mean Squared Error', 'Train Accuracy', 'Test Accuracy'])
        #firstHL = list(range(150,200))
        #secondHL = list(product(range(100,200),repeat=2))
        
        neurons = [10,50,100,150,200]
        thirdHL = list(product(neurons,repeat=3))
        hiddenLayers = thirdHL
        
        activations = ('relu', 'logistic')
        # Hidden Layers
        for i, act in enumerate(activations):
            for hl in hiddenLayers:
                #hl = i
                #currActivate = k
                #regpenalty = 0.0003 #according to forums on the internet, this is an optimal adam solver learning rate
                clf = MLPRegressor(hidden_layer_sizes=(hl)
                                    , activation=act
                                    , solver='adam'
                                    , alpha=0.04
                                    , max_iter=10000
                                    , validation_fraction=0.42).fit(X_train,y_train)
                annPredY = clf.predict(X_test)
                
                # Get  Scores
                train_accuracy = clf.score(X_train, y_train)
                test_accuracy = clf.score(X_test, y_test)
                mse = mean_squared_error(y_test, annPredY)
                
                print("\n ###### MPL Classifier #######")
                print(f"\n Activation Type: {act}")
                #print(f"\nLearning Rate: {learning_rate}")
                print(f"\nHidden Layers: {hl}")
                print("\n\rANN: MSE = %f" % mse)
                print(f"\nTrain Accuracy = {train_accuracy}")
                print(f"\nTest Accuracy = {test_accuracy}")
           
                new_row = {'Hidden Layer': hl, 
                            'Activation': act, 
                            'Mean Squared Error': mse,
                            'Train Accuracy': train_accuracy,
                            'Test Accuracy':test_accuracy}
                new_index = resultsDF.index.max()+1
                if i == 0:
                    new_index = 0
                resultsDF.loc[new_index] = new_row
        
        resultsDF.to_csv('MLP_Results.csv')
        t2 = datetime.now()
        total_time = t2-t1
        print("Started: ", t1)
        print("Ended: ", t2)
        print("Total Time for ANN: ", t2-t1)

    # Best Model
    clf = MLPRegressor(hidden_layer_sizes=(hl)
                        , activation=act
                        , solver='adam'
                        , alpha=0.04
                        , max_iter=10000
                        , validation_fraction=0.42).fit(X_train,y_train)
    
    print(f'R2: {clf.score(X_test, y_test)}')
    print(f'MSE: {mean_squared_error(y_test, clf.predict(X_test))}')
    

    if make_video:
        ittr_0_input_data = X[0:32,:-1]
        ittr_0_images = X[0:32,-1]
        ittr_0_target_data = y[0:32]
        model_predictions = clf.predict(ittr_0_input_data)
        make_prediction_video('mpl_ittr_0_prediction.mp4', ittr_0_images, ittr_0_target_data, model_predictions)
        
        
        ittr_1_input_data = X[32:64,:-1]
        ittr_1_images = X[32:64,-1]
        ittr_1_target_data = y[32:64]
        model_predictions = clf.predict(ittr_1_input_data)
        make_prediction_video('mpl_ittr_1_prediction.mp4', ittr_1_images, ittr_1_target_data, model_predictions)
        
       
        ittr_2_input_data = X[64:88,:-1]
        ittr_2_images = X[64:88,-1]
        ittr_2_target_data = y[64:88]
        model_predictions = clf.predict(ittr_2_input_data)
        make_prediction_video('mpl_ittr_2_prediction.mp4', ittr_2_images, ittr_2_target_data, model_predictions)
       
        ittr_3_input_data = X[88:122,:-1]
        ittr_3_images = X[88:122,-1]
        ittr_3_target_data = y[88:122]
        model_predictions = clf.predict(ittr_3_input_data)
        make_prediction_video('mpl_ittr_3_prediction.mp4', ittr_3_images, ittr_3_target_data, model_predictions)

#------------------------------------------------------------------------------

if __name__ == '__main__':
    run_linear_regression_model()
    run_MLPRegressor_model(train_model = True, make_video = False)
    
    