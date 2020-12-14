import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn import preprocessing

master = pd.read_csv("../save/y_train_cached.csv")

feature_columns = [c for c in master.columns if c.startswith("f_")]
target_columns = [c for c in master.columns if not c.startswith("f_")]
len(feature_columns)

X = master[feature_columns].drop(["f_hog", "f_red", "f_green", "f_blue"], axis=1)
y = master[target_columns].drop("file_name", axis=1)

scaler = preprocessing.MinMaxScaler().fit(X)
X_scaled = scaler.transform(X)


log = {
    "indoor": LogisticRegression(max_iter=1000).fit(X_scaled, y["indoor"]),
    "outdoor": LogisticRegression(max_iter=1000).fit(X_scaled, y["outdoor"]),
    "person": LogisticRegression(max_iter=1000).fit(X_scaled, y["person"]),
    "day": LogisticRegression(max_iter=1000).fit(X_scaled, y["day"]),
    "night": LogisticRegression(max_iter=1000).fit(X_scaled, y["night"]),
    "water": LogisticRegression(max_iter=1000).fit(X_scaled, y["water"]),
    "road": LogisticRegression(max_iter=1000).fit(X_scaled, y["road"]),
    "vegetation": LogisticRegression(max_iter=1000).fit(X_scaled, y["vegetation"]),
    "tree": LogisticRegression(max_iter=1000).fit(X_scaled, y["tree"]),
    "mountains": LogisticRegression(max_iter=1000).fit(X_scaled, y["mountains"]),
    "beach": LogisticRegression(max_iter=1000).fit(X_scaled, y["beach"]),
    "buildings": LogisticRegression(max_iter=1000).fit(X_scaled, y["buildings"]),
    "sky": LogisticRegression(max_iter=1000).fit(X_scaled, y["sky"]),
    "sunny": LogisticRegression(max_iter=1000).fit(X_scaled, y["sunny"]),
    "partly_cloudy": LogisticRegression(max_iter=1000).fit(X_scaled, y["partly_cloudy"]),
    "overcast": LogisticRegression(max_iter=1000).fit(X_scaled, y["overcast"]),
    "animal": LogisticRegression(max_iter=1000).fit(X_scaled, y["animal"]),
}

def prediction(input):
    print(input)
    input_resized = input.reshape(1, -1)
    # input_scaled = scaler.transform(input)
    pred = [model.predict(input_resized)[0] for model in log.values()]
    return pred

prediction(X_scaled[0])

# %%
import pathlib
import csv
s = "../data/test/23.jpg"
P = pathlib.Path(s)
P.name
a = [1, 2, 3]
b = list(map(str, a))
b
" ".join(b)
csv.writer(a)
pprint.pprint(a)


test = pd.read_csv("../save/y_test_cached.csv")
X_test = test[feature_columns + ["file_name"]].drop(["f_hog", "f_red", "f_green", "f_blue"], axis=1)
s = X_test.head(3)

#%%

def results_to_csv(rows):
    with open("../save/results.csv", "w") as file:
        for row in rows:
            path = pathlib.Path(row[-1])
            filename = row[-1]
            x_row = row[:-1]
            results = prediction(x_row)
            pp_results = " ".join(list(map(str, results)))
            file.write(f"{path.name} {pp_results}\n")

results_to_csv(X_test.values)
