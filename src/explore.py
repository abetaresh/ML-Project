import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

y_train = pd.read_csv("../data/training/training_labels.csv", sep=" ", header=0)
y_train.head()

y_summed = y_train.sum(axis=0)[1:]

fig, ax = plt.subplots()
# ax.set_xticks(np.arange(len(y_summed.index)))
# ax.set_xticklabels(y_summed.index, rotation=45)
plt.hist(y_summed)

# %%
import scipy.stats

percentage = y_summed / y_summed.sum()
series = np.array(percentage, dtype=np.float16)
scipy.stats.entropy(series)

series_uniform = np.array([1./17 for _ in range(17)])
scipy.stats.entropy(series_uniform)
