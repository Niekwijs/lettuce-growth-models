from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split

from data_processing import Data
from todo_remove_this_later_random_predictor import RandomYPredictor

# features = ["Variety", "Height", "Diameter", "LeafArea", "FreshWeightShoot", "DryWeightShoot"]
features = [ "Height", "Diameter", "LeafArea", "FreshWeightShoot", "DryWeightShoot"] # + ["Variety"] # Variety is not (yet) included in frontend

df = Data().tensor

X = df[features];
y = df[["Week"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# model = DummyRegressor() # Apparently this does not function as expected and does not return a "dummy" y value
model = RandomYPredictor(1,10)

model.fit(X_train, y_train)

tabular_predictor = model

if __name__ == "__main__":
    test_X = ['6205', '9847', '3924', '9580', '1701']
    for x in range(10):
        pred = model.predict(test_X)
        print(pred, len(pred))

