# %%
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import pathlib
import pandas as pd

# %%
print(tf.__version__)


data_dir = pathlib.Path("../data")
image_count = len(list(data_dir.glob("*/*.jpg")))
print(image_count)

# %% Labels
testing_labels  = pd.read_csv(data_dir/"testing/testing_labels.csv", sep=" ", header="infer")
training_labels = pd.read_csv(data_dir/"training/training_labels.csv", sep=" ", header="infer")
testing_labels.head()

images = list(data_dir.glob("testing/*.jpg"))
sample = images[0]
sample_encoding = testing_labels[testing_labels["file_name"] == sample.name] \
    .to_numpy()[0]
print(sample_encoding)

my_img = PIL.Image.open(str(images[0]))
my_img2 = PIL.Image.open(str(images[0]))
assert my_img.size == my_img2.size
print(my_img.size)
my_img2


# %% Test
batch_size = 32
img_height = 360
img_width = 480

# Remove file name
testing_labels = testing_labels.iloc[:, 1:].to_numpy()
training_labels = training_labels.iloc[:, 1:].to_numpy()

img_ds = tf.data.Dataset.list_files(str(data_dir/"training/*.jpg")) \
    .map(tf.io.read_file)
label_ds = tf.data.Dataset.from_tensor_slices(training_labels)
train_ds = tf.data.Dataset.zip((img_ds, label_ds))

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)



import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

grid = ImageGrid(fig, 111, nrows_ncols=(3,3), axes_pad=0.1)

%load_ext tensorboard
%tensorboard --logdir=/tmp/tensorflow_logs



a = list(img_ds.take(1).as_numpy_iterator())[0]
a
PIL.Image.open(a)
a[0]
type(a.astype("uint8"))

fig = plt.figure(figsize=(15,5))
for im, l in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        print(im[i])
        plt.imshow(im[i].numpy().astype("uint8"))
        plt.axis("off")

pd.options.display.html.table_schema = True
pd.options.display.max_rows = None

a = sns.load_dataset('iris')
a
