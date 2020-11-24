# %%
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
from conf import *

# %%
print(tf.__version__)

data_dir = pathlib.Path("../data")
image_count = len(list(data_dir.glob("*/*.jpg")))
print(image_count)

# %% Labeling

def parse_image(filename):
    # Reads the file into a string of bytes
    image = tf.io.read_file(filename)
    # Decodes the string into a Tensor
    image = tf.image.decode_jpeg(image)
    # Specifies the underlying data type
    image = tf.image.convert_image_dtype(image, tf.float32)
    # Unifies image input
    image = tf.image.resize(image, PARAMS["resize_dim"])
    return image

def read_labels(filepath):
    # Read meta data
    labels = pd.read_csv(filepath, sep=" ", header="infer")
    # Remove first "name" column eg. 23-00-11.jpg
    labels = labels.iloc[:, 1:].to_numpy()
    return labels

def build_ds(data_path, label_path):
    list_ds = tf.data.Dataset.list_files(str(data_path))
    img_ds = list_ds.map(parse_image)

    labels_np = read_labels(label_path)
    labels_ds = tf.data.Dataset.from_tensor_slices(labels_np)

    ds = tf.data.Dataset.zip((img_ds, labels_ds))
    return ds

train_ds = build_ds(data_dir/"training/*.jpg", data_dir/"testing/testing_labels.csv")
test_ds = build_ds(data_dir/"training/*.jpg", data_dir/"training/training_labels.csv")

# %% Visualizing checkpoint

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

fig = plt.figure(figsize=(12,9))
plt.axis("off")

grid = ImageGrid(fig, 111, nrows_ncols=(3,3), axes_pad=0.1)
for ax, (im, labels) in zip (grid, iter(train_ds.take(9))):
    ax.imshow(im)
    plt.axis("off")
plt.show()

# %% Tensorboard time?

# %% PCA
%load_ext tensorboard
%tensorboard --logdir=/tmp/tensorflow_logs

batch_size = 32

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)
