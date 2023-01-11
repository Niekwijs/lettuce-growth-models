import pandas as pd
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split

from Flask_API.data_processing import Data

data = Data()


features = ["Variety", "Height", "Diameter", "LeafArea", "FreshWeightShoot", "DryWeightShoot"]

df = data.read()

print(df.columns)

X_train = df[features];
y_train = df[["Week"]]
model = DummyClassifier()
model.fit(X_train, y_train)

tabular_predictor = model