import os
import netCDF4 as nc
import numpy as np
from PIL import Image
from datetime import datetime

from tqdm import tqdm

from termcolor import colored
import matplotlib.pyplot as plt

data_path = r"dataset\Mauritanie\CO03_mauritanie_plankton_hr_20220101_20220419_catsat_mauritanie "
base = data_path.split("\\")[1] + "\\" + data_path.split("\\")[2]
examples_path = os.path.join("examples", base)
train_db_path = os.path.join("database", base)
statistic_path = os.path.join("statistic", base)

for path in [examples_path, train_db_path, statistic_path]:
    os.makedirs(path, exist_ok=True)

land_mask_path = os.path.join(statistic_path, "land_mask.png")

#####################
LAND_MASK = True
CREATE_IMAGES = False
CREATE_ANIMATION = False
STATISTIC = False
CREATE_TRAINING_SET = True
CROP_COLUMN = 860
#####################


if LAND_MASK:
    print(colored("Create land mask...", color="blue", attrs=["bold"]))

    # Create land_mask initial zeros array
    ds = nc.Dataset(os.path.join(data_path, os.listdir(data_path)[0]))
    land_mask = np.zeros(ds.variables['chl'][0, :, :].shape)

    # Normalize and combine all frames
    for image in tqdm(os.listdir(data_path), ncols=70):
        ds = nc.Dataset(os.path.join(data_path, image))
        chl = np.array(ds.variables['chl'][0, :, :])

        chl /= chl.max() / 255.0
        land_mask += chl

    # Land mask: Combine all frames / Num(frames) --> Extract 255 pixels
    land_mask = land_mask // len(os.listdir(data_path))
    land_mask[land_mask != 255] = 0

    # Save land mask
    Image.fromarray(land_mask).convert("1").save(land_mask_path, mode="1")


if CREATE_IMAGES or CREATE_TRAINING_SET or STATISTIC:
    if CREATE_IMAGES:
        print(colored(f"Creating images...", color="blue", attrs=["bold"]))
    if CREATE_TRAINING_SET:
        print(colored(f"Creating training set...", color="blue", attrs=["bold"]))
    if STATISTIC:
        print(colored(f"Creating statistic...", color="blue", attrs=["bold"]))

    # Arrays for statistic collection
    clouds, dates = [], []

    # Calculate land cover
    land_mask = np.array(Image.open(land_mask_path))
    land_cover = round(land_mask.sum() / (land_mask.shape[0] * land_mask.shape[1]) * 100, 1)

    for file in tqdm(os.listdir(data_path), ncols=70):
        ds = nc.Dataset(os.path.join(data_path, file))

        # File features
        file_name = file.split(".")[0]
        file_date = datetime.strptime(file_name.split("_")[-2], '%Y%m%d').strftime('%d/%m/%Y')
        dates.append(file_date)

        # Extract mask data
        data = ds.variables['chl'][0, :, :]
        chl = np.array(data)
        chl /= chl.max() / 255.0

        # Calculate cloud cover: (data // 255).sum() - land_mask.sum()
        cloud_cover = round(((chl//255.0).sum() - land_mask.sum()) / (chl.shape[0] * chl.shape[1]) * 100, 1)
        clouds.append(cloud_cover)

        if CREATE_IMAGES:
            plt.figure(figsize=(13, 4))
            plt.suptitle(f"Date {file_date}\n\n\n", x=0.49, fontsize=13)

            plt.subplot(1, 2, 1)
            plt.figtext(.5, .87, f"Cloud cover: {cloud_cover}%, Land cover: {land_cover}%", fontsize=10, ha='center')

            # Create linear scale subplot
            plt.subplot(1, 2, 1)
            plt.imshow(data)
            plt.colorbar()
            plt.title("Linear scale", fontsize=10)

            # Create log scale subplot
            plt.subplot(1, 2, 2)
            plt.imshow(np.log10(data))
            plt.colorbar()
            plt.title("Log  scale", fontsize=10)

            # Save figure
            plt.subplots_adjust(wspace=0.07, top=0.81)
            plt.savefig(os.path.join(examples_path, file_name + "_double.png"))
            plt.close()

        if CREATE_TRAINING_SET:
            # Crop uninformative part
            if CROP_COLUMN:
                data = data[:, :CROP_COLUMN]

            log_data = np.log10(data)
            log_data[log_data == 0] = 0.01
            log_data[log_data == np.unique(log_data)[-1]] = 0
            np.save(os.path.join(train_db_path, f"{file_name}_{cloud_cover}.npy"), np.float16(np.array(log_data)))

    if STATISTIC:
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.bar(dates, clouds, bottom=land_cover, label='cloud cover')
        ax.bar(dates, land_cover, label='land cover')
        ax.axis(ymin=0, ymax=100, xmin=-0.7, xmax=len(os.listdir(data_path))-0.4)
        ax.legend()
        plt.xticks(dates[::2], rotation="vertical", fontsize=8)
        plt.title('The land and cloud cover')
        plt.savefig(os.path.join(statistic_path, "statistic.png"))
        plt.close()


if CREATE_ANIMATION:
    print(colored("Create animation...", color="blue", attrs=["bold"]))
    frames = []
    for file in os.listdir(examples_path):
        if file.endswith(".png"):
            frame = Image.open(os.path.join(examples_path, file))
            frames.append(frame)

    frames[0].save(os.path.join(examples_path, "a_history.gif"), save_all=True, append_images=frames[1:],
                   optimize=True, duration=100, loop=0)