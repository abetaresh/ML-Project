import pandas as pd
import tensorflow as tf
import pathlib

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

print("TF version: ", "\n\n" + tf.__version__)
print("Num GPUs available: ", len(tf.config.experimental.list_physical_devices("GPU")))

train_labels = pd.read_csv("../data/train/train_labels.csv", sep=' ', header=0)
data_dir = pathlib.Path("../data/train/")

batch_size = 32
img_width = 480
img_height = 320

train_df = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    labels="inferred",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)
