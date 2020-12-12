import numpy as np
import os
import PIL
import PIL.Image
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import yaml

# %% Loading global settings
with open("config.yml", "r") as config:
    SETTINGS = yaml.safe_load(config)

image_count = len(list(pathlib.Path("../data/").glob("**/*.jpg")))
print(image_count)

# %% Labeling

y_train = pd.read_csv("../data/train/train.csv", delimiter=" ", index_col="file_name")
y_test = pd.read_csv("../data/test/test.csv", delimiter=" ", index_col="file_name")
y_train.index = "../data/train/" + y_train.index
y_test.index = "../data/test/" + y_test.index
ys = pd.concat([y_train, y_test], ignore_index=False)

# %%
import matplotlib.image as mpimg

def show_sample(df, labels):
    df_queried = query(df, labels)
    sample = df_queried.sample()
    filename = sample.index[0]
    img = mpimg.imread(filename)
    plt.imshow(img)
    return sample

def query(df, labels):
    query = [l + " == 1" for l in labels]
    query = " & ".join(query)
    df_queried = df.query(query)
    return df_queried

indoors = y_train[y_train["indoor"] == 1]
indoors.sum(axis=0)

show_sample(ys, ["day", "mountains", "beach"])


#%%
label_weight = y_train[y_train == 1].sum()
plt.figure(figsize=(12, 5))
ax = label_weight.hist()
ax.set_xticklabels(label_weight.index)
ax.set(xlabel="class", ylabel="frequency")
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
plt.plot()

# %%
