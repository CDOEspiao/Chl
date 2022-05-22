import os
import numpy as np
from tqdm import tqdm
import imageio

years_for_dataset = ['2018', '2019', '2020', '2021', '2022']
area = 'Mauritanie'

train_db_path = os.path.join('database', area)

# Input: 5 frames, Predict: 1 frame
sequence_len = 6

# Read all data in years_for_dataset to training_paths
training_paths = []
for folder in os.listdir(train_db_path):
    for year in years_for_dataset:
        if year in folder:
            for file in os.listdir(os.path.join(train_db_path, folder)):
                training_paths.append(os.path.join(train_db_path, folder, file))

print(f"Database shape: {len(training_paths)}")

files_shape = np.load(training_paths[0]).shape
for i in tqdm(range(len(training_paths) - 1), ncols=70):
    if i == len(training_paths) - sequence_len + 1:
        break

    # Create sequence_len-frames numpy array
    training_sequence = np.zeros((sequence_len, files_shape[0], files_shape[1]))

    # Full sequence_len-frames numpy array
    for frame_num in range(sequence_len):
        training_sequence[frame_num, :, :] = np.load(training_paths[i+frame_num])

    # Add sequence to training base
    training_sequence = np.expand_dims(training_sequence, axis=0)
    if i == 0:
        training_base = training_sequence
    elif i % 10 == 0:
        np.save(f'training_base_{i}.npy', training_base)
        training_base = training_sequence
    else:
        training_base = np.concatenate([training_base, training_sequence], axis=0)

# np.save(f'training_base_{i}.npy', training_base)