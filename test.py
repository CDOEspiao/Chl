import numpy as np
import os
import sys
from tqdm import tqdm

database_folder = r"database/Mauritanie/CO03_mauritanie_plankton_hr_20180101_20181231_catsat_mauritanie"
database_folder_2 = r"database/Mauritanie/CO03_mauritanie_plankton_hr_20190101_20191231_catsat_mauritanie"


def float64to16(folder):
    for file in tqdm(os.listdir(folder)):
        file_path = os.path.join(folder, file)
        new_test_arr = np.float16(np.loadtxt(file_path))
        np.savetxt(file_path, new_test_arr)


def create_dataset(folder, dtype):
    for i, file in tqdm(enumerate(os.listdir(folder))):
        numpy_array = np.expand_dims(np.loadtxt(os.path.join(folder, file), dtype=dtype), axis=0)
        if i == 0:
            training_base = numpy_array
        else:
            training_base = np.concatenate([training_base, numpy_array], axis=0)
    return training_base





