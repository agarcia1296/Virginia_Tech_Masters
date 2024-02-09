
import pandas as pd
from tqdm import tqdm

def add_frame_count_to_df(df):
    ''' Adding Final Position to DF '''
    # Add addtinoal columns and initialize with NaN
    df['frame'] = float('NaN')   
    df['final_pos_x'] = float('NaN')
    df['final_pos_y'] = float('NaN')
    df['final_pos_z'] = float('NaN')
    
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
            full_ittr_indexes.append(index)
            last_value_x = df['curr_pos_x'][index]
            last_value_y = df['curr_pos_y'][index]
            last_value_z = df['curr_pos_z'][index]
            for n, idx in enumerate(full_ittr_indexes):
                df.at[idx, 'frame'] = n
                df.at[idx, 'final_pos_x'] = last_value_x
                df.at[idx, 'final_pos_y'] = last_value_y
                df.at[idx, 'final_pos_z'] = last_value_z
       
        # All other cases
        else:
            current_iteration = row['ittr']
            # When ittr doesn't change value
            if current_iteration == last_iteration:
                full_ittr_indexes.append(index)
            # When ittr changes value
            elif current_iteration != last_iteration:   
                last_value_x = df['curr_pos_x'][index-1]
                last_value_y = df['curr_pos_y'][index-1]
                last_value_z = df['curr_pos_z'][index-1]
                for n, idx in enumerate(full_ittr_indexes):
                    df.at[idx, 'frame'] = n
                    df.at[idx, 'final_pos_x'] = last_value_x
                    df.at[idx, 'final_pos_y'] = last_value_y
                    df.at[idx, 'final_pos_z'] = last_value_z
                full_ittr_indexes = []
                last_iteration = current_iteration
                full_ittr_indexes.append(index)
    return df

def get_previous_frame_value(df, iteration, frame, column, num_frames_before):
    # Get the frame number of the previous frame
    previous_frame = frame - num_frames_before
    
    # Check if the previous frame is within the same iteration
    if previous_frame >= 1:
        previous_value = df[(df['iteration'] == iteration) & (df['frame'] == previous_frame)][column].values
        if len(previous_value) > 0:
            return previous_value[0]
    
    # If the previous frame is not found in the same iteration, return NaN
    return float('NaN')

def add_previous_frames_data(df):
    ''' Add data from 1 2 and 3 frames before for rows where the data is not NaN '''
    for i in range(1,4):
        for col in ['velocity_x', 'velocity_y', 'velocity_z', 'curr_pos_x', 'curr_pos_y', 'curr_pos_z']:
            new_col_name = f'{col}_{i}framesbefore'
            df[new_col_name] = df[col].shift(i)
    
    # Itterate over rows
    for index, row in tqdm(df.iterrows()):
        frame = row['frame']
        ittr = row['ittr']
        frame_0_of_ittr = df[(df['ittr'] == ittr) & (df['frame'] == 0)]
        # First Frame
        if frame == 0:
            for i in range(1,4):
                for col in ['velocity_x', 'velocity_y', 'velocity_z', 'curr_pos_x', 'curr_pos_y', 'curr_pos_z']:
                    new_col_name = f'{col}_{i}framesbefore'
                    row[new_col_name] = row[col]
                    df.iloc[index] = row
        # Second Frame
        if frame == 1:
            for i in range(2,4):
                for col in ['velocity_x', 'velocity_y', 'velocity_z', 'curr_pos_x', 'curr_pos_y', 'curr_pos_z']:
                    new_col_name = f'{col}_{i}framesbefore'
                    row[new_col_name] = frame_0_of_ittr[col].to_numpy()
                    df.iloc[index] = row
        # Third Frame
        if frame == 2:
            for i in range(3,4):
                for col in ['velocity_x', 'velocity_y', 'velocity_z', 'curr_pos_x', 'curr_pos_y', 'curr_pos_z']:
                    new_col_name = f'{col}_{i}framesbefore'
                    row[new_col_name] = frame_0_of_ittr[col].to_numpy()
                    df.iloc[index] = row
        
    return df


df = pd.read_csv('training_data/ittr_df.csv', index_col=0)
df = add_frame_count_to_df(df)
df = df.drop(columns = ['force_x', 'force_y', 'force_z'])
#df_filtered = df[df['frame'] > 3]

df_with_previous_frames_data = add_previous_frames_data(df)
df_with_previous_frames_data['image_path'] = df_with_previous_frames_data.apply(lambda row: f"training_data/ittr{int(row['ittr']):03}/frame_{int(row['frame']):03}.jpg", axis=1)
df_with_previous_frames_data.to_csv('training_data/FINAL_DF_WITH_PREVIOUS_DATA.csv')


