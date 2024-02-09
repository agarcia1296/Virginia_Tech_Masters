# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 02:29:06 2024

@author: agarc
"""
import pandas as pd
from tqdm import tqdm

def add_frame_count_to_df(df):
    ''' Adding Final Position to DF '''
    # Add 'last value' column and initialize with NaN
    df['frame'] = float('NaN')
    #df['final_pos_x'] = float('NaN')
    #df['final_pos_y'] = float('NaN')
    #df['final_pos_z'] = float('NaN')
    
    # Initialize vars
    full_ittr_indexes = []
    
    # Itterate over rows
    for index, row in tqdm(df.iterrows()):
        # First Index
        if index == 0:
            last_iteration = 0
            full_ittr_indexes.append(index)
            
        # Last Index
        elif index == df.index[-1]:
            current_iteration = row['ittr']
            #last_value_x = df['curr_pos_x'][index]
            #last_value_y = df['curr_pos_y'][index]
            #last_value_z = df['curr_pos_z'][index]
            for n, idx in enumerate(full_ittr_indexes):
                df.at[idx, 'frame'] = n
                #df.at[idx, 'final_pos_x'] = last_value_x
                #f.at[idx, 'final_pos_y'] = last_value_y
                #df.at[idx, 'final_pos_z'] = last_value_z
       
        # All other cases
        else:
            current_iteration = row['ittr']
            # When ittr doesn't change value
            if current_iteration == last_iteration:
                full_ittr_indexes.append(index)
            # When ittr changes value
            elif current_iteration != last_iteration:   
                #last_value_x = df['curr_pos_x'][index-1]
                #last_value_y = df['curr_pos_y'][index-1]
                #last_value_z = df['curr_pos_z'][index-1]
                for n, idx in enumerate(full_ittr_indexes):
                    df.at[idx, 'frame'] = n
                    #df.at[idx, 'final_pos_x'] = last_value_x
                    #df.at[idx, 'final_pos_y'] = last_value_y
                    #df.at[idx, 'final_pos_z'] = last_value_z
                full_ittr_indexes = []
                last_iteration = current_iteration
                full_ittr_indexes.append(index)
    return df


df = pd.read_csv('training_data/ittr_df.csv', index_col=0)
df = add_frame_count_to_df(df)

#------------------------------------------------------------------------------
#%% Get all ittr to be same size

# Find the indices where the sequence restarts to 0
restart_indices = df[df['frame'] == 0].index.tolist()

# Calculate the smallest number of frames before the sequence restarts
smallest_frames = float('inf')  # Initialize with infinity
for i in range(1, len(restart_indices)):
    consecutive_frames = restart_indices[i] - restart_indices[i-1] - 1
    if consecutive_frames < smallest_frames:
        smallest_frames = consecutive_frames

print("Smallest number of frames before restart:", smallest_frames)

desired_frames = smallest_frames

for ittr in tqdm(range(df['ittr'].max() + 1)):
    # DF for all rows in same ittr
    filtered_df = df[df['ittr'] == ittr]
    frames_to_remove = len(filtered_df) - desired_frames
    
    # Don't do anything if no frames need to be removed
    if frames_to_remove == 0:
        continue
    
    # Remove every odd index until total frames dropped is met
    total_removed = 0
    for index_to_drop in filtered_df.index:
        if index_to_drop % 2 == 1:
            continue
        elif index_to_drop % 1 != 1:
            df = df.drop(index_to_drop)
            total_removed += 1
            if total_removed == frames_to_remove:
                break
df = df.reset_index(drop = True)
#df = add_frame_count_to_df(df)

#%% Get all ittr in one row

changing_columns = ['frame', 'velocity_x', 'velocity_y', 'velocity_z', 'curr_pos_x', 'curr_pos_y', 'curr_pos_z']
non_changing_columns = ['ittr', 'force_x','force_y', 'force_z']
df_final = pd.DataFrame(columns = non_changing_columns)

columns = non_changing_columns.copy()
for index in range(21):
    numbered_columns_list = [value + f'_{index}' for value in changing_columns]
    columns.extend(numbered_columns_list)

df_final = pd.DataFrame(columns = columns)

for row, ittr in tqdm(enumerate(range(df['ittr'].max() + 1))):
    # DF for all rows in same ittr
    filtered_df = df[df['ittr'] == ittr]
    first_index = filtered_df.index[0]
    init_df = filtered_df.loc[first_index, non_changing_columns].tolist()  # Extract first row of non-changing columns
    for index in filtered_df.index:
        frame_data = filtered_df.loc[index, changing_columns].tolist()
        init_df.extend(frame_data)
        #frame_data = frame_data.rename(index=dict(zip(frame_data.index, numbered_columns_list)))
        #df_final.loc[row] = frame_data
    
    df_final.loc[row] = init_df

df_final.to_csv('training_data/FINAL_DF.csv')
