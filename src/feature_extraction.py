import tensorflow as tf
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skimage.io import imread, imshow
from start import y_train, y_test
import plotly.express as px
from sklearn.linear_model import LogisticRegression
import skimage.feature
from skimage.feature.corner import corner_harris, corner_subpix
from skimage.color import rgb2gray
import skimage
from sklearn.decomposition import PCA

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

im = imread(y_train.index[0])
im2 = imread(y_train.index[9])
a, b, = skimage.feature.hog(im, orientations=9, visualize=True)
a.shape, b.shape
im.shape
skimage.feature.hog(im2).shape

# %% corner detection with subpix 

def corner_detection(filepath):
	img = skimage.io.imread(filepath, as_gray = True)
	
	# Creates the edgse
	corners =  peak_local_max(corner_harris(img), min_distance=10, threshold_rel=0, num_peaks=1)
	#print(corners)
    
	subpix_edges = skimage.feature.corner_subpix(img, corners)

	# Discards nan values
	nb_of_edges = np.count_nonzero(~np.isnan(subpix_edges))
	return nb_of_edges


# Maps images to their number of corners
y_train.feature_corners = list(map(corner_detection, y_train.index))
where y_train.index is ["../data/train/sample.jpg", ...]



# %% SVM
image = map(imread, y_train.index)
next(image)

# %%
import skimage.io
i = imread(y_train.index[0])

skimage.io.imsave("../res/random_image.png", i)
imshow(i)
