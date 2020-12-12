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
    df["f_lum"] = list(map(lambda x: np.mean(imread(x)), "../data/train/" + df.index))
    return df

y_train = luminosity(y_train)

# %%
px.scatter_3d(y_train, x="day", y="night", z="f_lum")

# %%

X = y_train[["f_lum"]]
y = y_train[["day", "night"]]
reg = LogisticRegression().fit(X, y)
reg.score(X, y)

# %%
y_train.head()
