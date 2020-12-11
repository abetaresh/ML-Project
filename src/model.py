import tensorflow as tf
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skimage.io import imread, imshow
from start import y_train, y_test

# %%
# Idea 1: Train separate models on the groups
label_group = [
    ["indoor", "outdoor"],
    ["day", "night"],
    ["sunny", "partly_cloudy", "overcast"],
]


# %%
from sklearn.linear_model import LinearRegression
def luminosity(df):
    df["f_lum"] = list(map(lambda x: np.mean(imread(x)), "../data/train/" + df.index))
    return df

y_train = luminosity(y_train)

# %%
import plotly.express as px
px.scatter_3d(y_train, x="day", y="night", z="f_lum")

# %%

X = y_train[["f_lum"]]
y = y_train[["day", "night"]]
reg = LinearRegression().fit(X, y)
reg.score(X, y)

# %%
y_train.head()
