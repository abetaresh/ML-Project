from start import y_train, y_test
import numpy as np
import skimage
from skimage.io import imread, imshow
import plotly.express as px
import skimage.feature
from skimage.feature.corner import corner_harris, corner_subpix
from skimage.feature import peak_local_max
from skimage.color import rgb2gray
import plotly.express as px
import skimage

# %%
# Idea 1: Train separate models on the groups
label_group = [
    ["indoor", "outdoor"],
    ["day", "night"],
    ["sunny", "partly_cloudy", "overcast"],
]


# %%
def luminosity(df):
    df["f_lum"] = list(map(lambda x: np.mean(imread(x)), df.index))

def min_point(df):
    df["f_min"] = list(map(lambda im: np.min(imread(im)), df.index))

def red(df):
    def colorify(filepath):
        img = imread(filepath)
        R = img[:, :, 0]
        R_pad = np.concatenate([R.flatten(), np.arange(0, 256)])
        _, R_count = np.unique(R_pad, return_counts=True)
        # Normalize
        R_count = R_count / R_count.sum()
        series = {"f_red" + str(i): x for i, x in enumerate(R_count.flatten())}
        return series

    df["f_red"] = list(map(colorify, df.index))

def green(df):
    def colorify(filepath):
        img = imread(filepath)
        G = img[:, :, 1]
        G_pad = np.concatenate([G.flatten(), np.arange(0, 256)])
        _, G_count = np.unique(G_pad, return_counts=True)
        # Normalize
        G_count = G_count / G_count.sum()
        series = {"f_green" + str(i): x for i, x in enumerate(G_count.flatten())}
        return series

    df["f_green"] = list(map(colorify, df.index))

def blue(df):
    def colorify(filepath):
        img = imread(filepath)
        B = img[:, :, 0]
        B_pad = np.concatenate([B.flatten(), np.arange(0, 256)])
        _, B_count = np.unique(B_pad, return_counts=True)
        # Normalize
        B_count = B_count / B_count.sum()
        series = {"f_blue" + str(i): x for i, x in enumerate(B_count.flatten())}
        return series

    df["f_blue"] = list(map(colorify, df.index))

def hog(df):
    def hogify(filepath):
        img = imread(filepath)
        feature = skimage.feature.hog(img,
                                      orientations=9,
                                      pixels_per_cell=(50, 50),
                                      visualize=False)
        feature = np.pad(feature, (0, 3000 - len(feature)), mode="constant")
        series = {"f_hog" + str(i): x for i, x in enumerate(feature)}
        return series

    df["f_hog"] = list(map(hogify, df.index))

def corners(df):
    def cornerify(filepath):
        img = imread(filepath)
        grayscale = rgb2gray(img)
        harris = corner_harris(grayscale)
        coords = peak_local_max(harris, min_distance=20, num_peaks=50)
        corners_subpix = corner_subpix(grayscale, coords)
        return corners_subpix.flatten()

    df["f_corners"] = list(map(cornerify, df.index))

luminosity(y_train)
print("LUMINOSITY")
min_point(y_train)
print("MIN")
hog(y_train)
print("HOG")
red(y_train)
print("RED")
green(y_train)
print("GREEN")
blue(y_train)
print("BLUE")

unpack_hog = y_train.f_hog.apply(pd.Series)
unpack_red = y_train.f_red.apply(pd.Series)
unpack_green = y_train.f_green.apply(pd.Series)
unpack_blue = y_train.f_blue.apply(pd.Series)

y_train = pd.concat([y_train, unpack_hog, unpack_red, unpack_green, unpack_blue], axis=1)

y_train.to_csv("../save/y_train_cached.csv")

# %%

luminosity(y_test)
print("LUMINOSITY")
min_point(y_test)
print("MIN")
hog(y_test)
print("HOG")
red(y_test)
print("RED")
green(y_test)
print("GREEN")
blue(y_test)
print("BLUE")

unpack_hog = y_test.f_hog.apply(pd.Series)
unpack_red = y_test.f_red.apply(pd.Series)
unpack_green = y_test.f_green.apply(pd.Series)
unpack_blue = y_test.f_blue.apply(pd.Series)

y_test = pd.concat([y_test, unpack_hog, unpack_red, unpack_green, unpack_blue], axis=1)

y_test.to_csv("../save/y_test_cached.csv")


#%%
import pandas as pd

y_train.head(1)
hogs = y_train.f_hog.apply(pd.Series)
pd.concat([y_train, hogs], axis=1)

# %%
# px.scatter_3d(y_train, x="day", y="night", z="f_lum")
# px.scatter(y_train, x="f_min", y="day")
