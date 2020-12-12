import tensorflow as tf
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skimage.io import imread, imshow
from start import y_train, y_test
import plotly.express as px

# %%
# Idea 1: Train separate models on the groups
label_group = [
    ["indoor", "outdoor"],
    ["day", "night"],
    ["sunny", "partly_cloudy", "overcast"],
]


# %%
from sklearn.linear_model import LogisticRegression
def luminosity(df):
    df["f_lum"] = list(map(lambda x: np.mean(imread(x)), df.index))
    return df

y_train = luminosity(y_train)
y_train.to_csv("chached.csv")
# %%
px.scatter_3d(y_train, x="day", y="night", z="f_lum")

# %% Logistic Regression on luminosity to predict the time of day
X = y_train[["f_lum"]]
reg_day = LogisticRegression().fit(X, y_train.day)
reg_night = LogisticRegression().fit(X, y_train.night)
reg_day.score(X, y_train.day)
reg_night.score(X, y_train.night)

# %% HOG: Histogram of Oriented Gradients
import skimage.feature
import skimage

im = imread("../data/train/" + y_train.index[0])
im2 = imread("../data/train/" + y_train.index[9])
a, b, = skimage.feature.hog(im, orientations=9, visualize=True)
a.shape, b.shape
im.shape
skimage.feature.hog(im2).shape

# %% SVM
from sklearn.decomposition import PCA
image = map(imread, y_train.index)
next(image)
