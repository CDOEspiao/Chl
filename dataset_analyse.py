import os
import netCDF4 as nc
import numpy as np
from PIL import Image
from datetime import datetime

from tqdm import tqdm

from termcolor import colored
import matplotlib.pyplot as plt
from numba import jit

data_path = r"dataset\Mauritanie\CO03_mauritanie_plankton_hr_20220101_20220419_catsat_mauritanie"
base = data_path.split("\\")[1] + "\\" + data_path.split("\\")[2]
examples_path = os.path.join("examples", base)
train_db_path = os.path.join("database", base)
statistic_path = os.path.join("statistic", base)

for path in [examples_path, train_db_path, statistic_path]:
    os.makedirs(path, exist_ok=True)


#####################
CREATE_IMAGES = True
CREATE_TRAINING_SET = True
CREATE_ANIMATION = True
TERRA_MASK = False
STATISTIC = True
#####################

@jit
def calculate_cloud_cover(mask, arr, mask_value, cover=0):
    for row in range(chl.shape[0]):
        for col in range(chl.shape[1]):
            if ((mask[row][col] != 255) and (arr[row][col] == mask_value)):
                cover += 1
    return cover


if TERRA_MASK:
    print(colored("Create terra mask...", color="blue", attrs=["bold"]))
    img = Image.open(os.path.join(train_db_path, os.listdir(train_db_path)[0]))
    terra_mask = np.zeros((np.array(img).shape))
    for i, image in enumerate(tqdm(os.listdir(train_db_path), ncols=70)):
        img = Image.open(os.path.join(train_db_path, image))
        np_arr = np.array(img)
        # unique = np.unique(np_arr)
        # print(f"Shape: {np_arr.shape}\tUnique: {unique}")
        terra_mask += np_arr

    terra_mask = terra_mask // (i+1)
    terra_mask[terra_mask != 255] = 0
    Image.fromarray(terra_mask).convert("L").save(os.path.join(statistic_path, "terra_mask.png"), mode="L")

if CREATE_IMAGES or CREATE_TRAINING_SET or STATISTIC:
    if CREATE_IMAGES:
        print(colored(f"Creating images...", color="blue", attrs=["bold"]))
    if CREATE_TRAINING_SET:
        print(colored(f"Creating training set...", color="blue", attrs=["bold"]))
    if STATISTIC:
        print(colored(f"Creating statistic...", color="blue", attrs=["bold"]))

    clouds, dates = [], []
    ds = nc.Dataset(os.path.join(data_path, os.listdir(data_path)[0]))

    frame_size = ds.variables['chl'][0, :, :].shape
    terra_mask = np.array(Image.open(os.path.join(statistic_path, "terra_mask.png")))
    terra_pixels = terra_mask.sum()//255
    terra_perc = round(terra_pixels / (frame_size[0] * frame_size[1]) * 100, 1)

    for i, file in enumerate(tqdm(os.listdir(data_path), ncols=70)):
        ds = nc.Dataset(os.path.join(data_path, file))
        filename = file.split(".")[0]

        date = filename.split("_")[-2]
        date = datetime.strptime(date, '%Y%m%d').strftime('%d/%m/%Y')
        dates.append(date)

        data = ds.variables['chl']
        lat, lon = ds.variables['latitude'][:], ds.variables['longitude'][:]
        chl = ds.variables['chl'][0, :, :]

        norm = np.array(Image.fromarray(chl).convert("L"))

        # Calculate cloud cover
        cloud_pixels = calculate_cloud_cover(terra_mask, norm, 255)
        cloud_cover = round(cloud_pixels / (frame_size[0] * frame_size[1]) * 100, 1)
        clouds.append(cloud_cover)

        if CREATE_IMAGES:
            plt.contourf(chl)
            plt.suptitle(f"Date {date}", x=0.45)
            plt.title(f"Cloud cover: {cloud_cover}%, Terra cover: {terra_perc}%", fontsize=10)
            plt.colorbar().set_label(ds.variables['chl'].units)
            plt.savefig(os.path.join(examples_path, filename+".png"))
            plt.close()

        if CREATE_TRAINING_SET:
            Image.fromarray(np.array(chl)).convert("L").save(os.path.join(train_db_path, filename+"_{}_.png".format\
                                                                            (str(cloud_cover).split(".")[0])), mode="L")

if STATISTIC:
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.bar(dates, clouds, bottom=terra_perc, label='cloud cover')
    ax.bar(dates, terra_perc, label='land cover')
    ax.axis(ymin=0, ymax=100, xmin=-0.7, xmax=len(os.listdir(data_path))-0.4)
    ax.legend()
    plt.xticks(dates[::2], rotation="vertical", fontsize=10)
    plt.title('The land and cloud cover')
    plt.savefig(os.path.join(statistic_path, "statistic.png"))

if CREATE_ANIMATION:
    print(colored("Create animation...", color="blue", attrs=["bold"]))
    frames = []
    for file in os.listdir(examples_path):
        if file.endswith(".png"):
            frame = Image.open(os.path.join(examples_path, file))
            frames.append(frame)

    frames[0].save(os.path.join(examples_path, "a_history.gif"), save_all=True, append_images=frames[1:],
                   optimize=True, duration=100, loop=0)
