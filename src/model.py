import tensorflow as tf
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from start import train_ds, test_ds

# %%
import sklearn
import sklearn.svm

img_ds = train_ds.map(lambda img, label: img)
label_ds = train_ds.map(lambda img, label: label)

img_ds

clf = make_pipeline(StandardScaler, sklearn.svm.SVC(gamma="auto"))
clf.fit(img_ds, label_ds)
