
import pandas as pd
import numpy as np
import os
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropput

# gpus = tf.config.experimental.list_logical_devices('GPU')
# print(gpus)
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)




# Define CNN branch
class CNNBranch(nn.Module):
    def __init__(self):
        super(CNNBranch, self).__init__()
        self.cnn = models.resnet18(pretrained=True)
        # Modify the last layer to match your task
        num_features = self.cnn.fc.in_features
        self.cnn.fc = nn.Linear(num_features, 256)  # Example modification

    def forward(self, x):
        x = self.cnn(x)
        return x

# Define ML model branch
class MLModelBranch(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLModelBranch, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CombinedModel(nn.Module):
    def __init__(self, output_size, num_metadata_features, hidden_size):
        super(CombinedModel, self).__init__()
        # Image CNN branch
        self.cnn_branch = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            # Add more layers as needed
        )
        # Metadata FNN branch
        self.metadata_fnn = MLModelBranch(num_metadata_features, hidden_size)

    def forward(self, image_data, metadata):
        # Forward pass through the image CNN branch
        cnn_output = self.cnn_branch(image_data)
        # Forward pass through the metadata FNN branch
        metadata_output = self.metadata_fnn(metadata)
        # Flatten the CNN output
        cnn_output_flat = cnn_output.view(cnn_output.size(0), -1)
        # Concatenate the flattened CNN output with the metadata output
        combined_output = torch.cat((cnn_output_flat, metadata_output), dim=1)
        return combined_output

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

def main():
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
    
    #--------------------------------------------------------------------------
    
    from sklearn.linear_model import LinearRegression
    
    linreg_model = LinearRegression().fit(metadata_train, y_train)
    linreg_model.score(metadata_test, y_test)
    
    linreg_model.predict(metadata_test)
   
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
    from sklearn.neural_network import MLPRegressor
    
    mlp_regressor = MLPRegressor(hidden_layer_sizes=(100, 50), 
                                 activation='relu', 
                                 solver='adam', 
                                 max_iter=500, 
                                 random_state=42)

    mlp_regressor.fit(metadata_train, y_train)
    mlp_regressor.score(metadata_test, y_test)

    
    ittr_0_input_data = X[0:32,:-1]
    ittr_0_images = X[0:32,-1]
    ittr_0_target_data = y[0:32]
    model_predictions = mlp_regressor.predict(ittr_0_input_data)
    make_prediction_video('mpl_ittr_0_prediction.mp4', ittr_0_images, ittr_0_target_data, model_predictions)
    
    
    ittr_1_input_data = X[32:64,:-1]
    ittr_1_images = X[32:64,-1]
    ittr_1_target_data = y[32:64]
    model_predictions = mlp_regressor.predict(ittr_1_input_data)
    make_prediction_video('mpl_ittr_1_prediction.mp4', ittr_1_images, ittr_1_target_data, model_predictions)
    
   
    ittr_2_input_data = X[64:88,:-1]
    ittr_2_images = X[64:88,-1]
    ittr_2_target_data = y[64:88]
    model_predictions = mlp_regressor.predict(ittr_2_input_data)
    make_prediction_video('mpl_ittr_2_prediction.mp4', ittr_2_images, ittr_2_target_data, model_predictions)
   
    ittr_3_input_data = X[88:122,:-1]
    ittr_3_images = X[88:122,-1]
    ittr_3_target_data = y[88:122]
    model_predictions = mlp_regressor.predict(ittr_3_input_data)
    make_prediction_video('mpl_ittr_3_prediction.mp4', ittr_3_images, ittr_3_target_data, model_predictions)

#------------------------------------------------------------------------------

if __name__ == '__main__':
    main()
    
    
#%%
cnn_branch = CNNBranch()
ml_model_branch = MLModelBranch(input_size=num_metadata_features, hidden_size=64, output_size=len(target_columns))
combined_model = CombinedModel(cnn_branch, ml_model_branch, combined_hidden_size=128)


# Define your data loaders
train_dataset = TensorDataset(img_data_tensor_train, meta_data_tensor_train, targets_train)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_dataset = TensorDataset(img_data_tensor_test, meta_data_tensor_test, targets_test)
val_loader = DataLoader(val_dataset, batch_size=1)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(combined_model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    # Training
    combined_model.train()
    for images, data_arrays, labels in train_loader:
        optimizer.zero_grad()
        outputs = combined_model(images, data_arrays)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validation
    combined_model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, data_arrays, labels in val_loader:
            outputs = combined_model(images, data_arrays)
            val_loss += criterion(outputs, labels).item()
    val_loss /= len(val_loader)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {val_loss:.4f}")