
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import os
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------

class ImageCNN(nn.Module):
    def __init__(self, output_size=1):
        super(ImageCNN, self).__init__()
        # Load a pre-trained CNN model (e.g., ResNet-18)
        self.cnn = models.resnet18(pretrained=True)
        # Modify the last layer to output a single continuous value
        num_features = self.cnn.fc.in_features
        self.cnn.fc = nn.Linear(num_features, output_size)

    def forward(self, x):
        # Forward pass through the CNN
        x = self.cnn(x)
        return x

#------------------------------------------------------------------------------

class MetadataFNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MetadataFNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)  # Single output neuron for regression

    def forward(self, x):
        # Forward pass through the FNN
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

#------------------------------------------------------------------------------

class CombinedModel(nn.Module):
    def __init__(self, output_size, num_metadata_features, hidden_size):
        super(CombinedModel, self).__init__()
        # Image CNN
        self.image_cnn = ImageCNN(output_size=1)
        # Metadata FNN
        self.metadata_fnn = MetadataFNN(num_metadata_features, hidden_size)

    def forward(self, image_data, metadata):
        # Forward pass through the image CNN
        image_output = self.image_cnn(image_data)
        # Forward pass through the metadata FNN
        metadata_output = self.metadata_fnn(metadata)
        # Combine the outputs
        combined_output = image_output + metadata_output  # You can adjust how the outputs are combined
        return combined_output

#------------------------------------------------------------------------------

def open_all_images(ittr_path:list):
    # Create cv objects of each image
    images_list = []
    for i, folder in tqdm(enumerate(ittr_path), desc = 'Opening Images'):
        img_list = []
        # Get list of all filenames in folder
        file_list = os.listdir(folder)
        image_files = [f for f in file_list if f.lower().endswith(('.jpg'))]
        image_files = image_files[:-1] # Drops zero_image.jpg
        # Sort the filenames based on the numerical part
        #sorted_filenames = sorted(image_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
        # Cut the list in half
        #half_length = len(sorted_filenames) // 2 //2
        #first_half = sorted_filenames[:half_length]
        # Remove every other item 3 times
        #first_half_image_files = first_half[::2][::2][::2]
        # Loop through the image files and read them into OpenCV objects
        for image_file in image_files:
            image_path = os.path.join(folder, image_file)
            image = cv2.imread(image_path)
            img_list.append(image)
        # Append this to the overall list
        images_list.append(img_list)
    return images_list

#------------------------------------------------------------------------------

# Now process the training and testing videos separately
def process_videos(ittr_dir:list, frame_nums:list, metadata:list):
    video_data = []
    for ittr, frame_list, meta in tqdm(zip(ittr_dir, frame_nums, metadata), desc = "Processing Videos"):
        # Make list of files frames
        file_list = []
        for file in os.listdir(ittr):
            for frame in frame_list:
                if f"{int(frame):03}" in file:
                    file_list.append(file)
                    
        video_frames = [cv2.cvtColor(cv2.imread(os.path.join(ittr, im_path)), cv2.COLOR_BGR2GRAY) for im_path in file_list]
        # Assuming each video has a fixed number of frames (e.g., 20 frames)
        video_frames = torch.tensor(video_frames)  # Convert to tensor if necessary

        # Extract metadata for each frame
        num_frames = len(frame_list)
        meta_tensor = []
        for i in range(num_frames):
            meta_start_index = i * 7
            meta_end_index = (i + 1) * 7
            meta_frame = meta[meta_start_index:meta_end_index]  # Extract metadata for current frame
            meta_frame = torch.tensor(meta_frame.astype(float), dtype=torch.float32)
            meta_tensor.append(meta_frame)

        # Concatenate metadata tensors along the channel dimension
        meta_tensor = torch.stack(meta_tensor, dim=0)

        # Expand metadata tensor to match the spatial dimensions of video frames
        meta_tensor = meta_tensor.unsqueeze(-1).unsqueeze(-1).expand(video_frames.shape[:2] + (-1, -1))

        # Concatenate video frames with metadata
        combined_tensor = torch.cat((video_frames, meta_tensor), dim=3)  # Concatenate along the channel dimension
            
        video_data.append(combined_tensor)  # Append combined tensor to video_data
            
    return torch.stack(video_data, dim=0)  # Stack video data into a tensor

#------------------------------------------------------------------------------

def main():
    # Import Meta Data
    meta_df = pd.read_csv('training_data/FINAL_DF_WITH_PREVIOUS_DATA.csv', index_col=0)
    
    # Add ittr path to df
    ittr_path = []
    for i in range(1000):
        ittr_path.append(f'training_data/ittr{i:03}')
    meta_df['ittr_path'] = ittr_path
    
    # Add Frame List
    frame_cols = [x for x in meta_df.columns if 'frame' in x]
    frame_df = meta_df.drop(columns=meta_df.columns.difference(frame_cols))
    meta_frames_col = [row.to_list() for idx, row in frame_df.iterrows()]
    meta_df['frames_list'] = meta_frames_col
    
    # Split into Test and Train data
    target_columns = ['final_pos_x', 'final_pos_y', 'final_pos_z']
    X = meta_df.drop(columns = target_columns).to_numpy()
    num_metadata_features = X.shape[1]
    y = meta_df.drop(columns=meta_df.columns.difference(target_columns)).to_numpy()
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Load video frames and process them
    ittr_dir_train = X_train[:, -2]
    frame_nums_train =  X_train[:, -1]
    metadata_train =  X_train[:,4:-2]
    video_data_train = process_videos(ittr_dir_train, frame_nums_train, metadata_train)  # Video Paths, Frames, meta data
    
    ittr_dir_test = X_test[:, -2]
    frame_nums_test =  X_test[:, -1]
    metadata_test =  X_test[:,4:-2]
    video_data_test = process_videos(ittr_dir_test, frame_nums_test, metadata_test)
    
    # Now y_train contains the target values for your training data
    targets_train = torch.tensor(y_train, dtype=torch.float32)
    targets_test = torch.tensor(y_test, dtype=torch.float32)
    
    # Define the model
    model = CombinedModel(output_size=3, num_metadata_features=num_metadata_features, hidden_size=64)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Define a PyTorch Dataset
    train_dataset = TensorDataset(video_data_train, targets_train)
    
    # Define batch size
    batch_size = 16  # Adjust according to your memory constraints
    
    # Create DataLoader instance
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    
    # Training loop
    for epoch in range(50):
        for batch in train_dataloader:  # Iterate over batches
            images, metadata, targets = batch    
            # Zero the gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(images, metadata)
            # Compute the loss
            loss = criterion(outputs, targets)
            # Backward pass
            loss.backward()
            # Update weights
            optimizer.step()
            
    
    
    # Example input tensors (adjust according to your data)
    image_data = torch.randn(1, 3, 224, 224)  # Example: batch size 1, 3 channels (RGB), 224x224 image size
    metadata = torch.randn(1, 4)  # Example: batch size 1, 4 metadata features
    
    # Forward pass
    output = model(image_data, metadata)
    print("Output shape:", output.shape)

#------------------------------------------------------------------------------

if __name__ == '__main__':
    main()