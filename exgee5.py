# Import the necessary libraries
import pandas as pd
import xgboost as xgb
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Read in the data
f = open("./data/measurements.json")
data_json = json.loads(f.read())["Measurements"]
vs = data_json.values()
df = pd.json_normalize(vs)

# Add an "ID" column by parsing it from the "RGB_Image" column
df["ID"] = df["RGB_Image"].str.strip(".png").str.strip("RGB_").apply(int)

prev = ""
counter = 0
def add_week_callback(item):
    global counter
    global prev
    if prev != item["Variety"]:
        counter += 1
    prev = item.Variety
    return counter
df["Week"] = df.apply(func=add_week_callback, axis=1)

# Encode the categorical variables as numerical values
le = LabelEncoder()
df['Variety'] = le.fit_transform(df['Variety'])
df['RGB_Image'] = le.fit_transform(df['RGB_Image'])
df['Depth_Information'] = le.fit_transform(df['Depth_Information'])
df['Week'] = le.fit_transform(df['Week'])

# Create the target column
df['FullyGrown'] = (df['Diameter'] >= 15).astype(int)

# Split the data into training and test sets
X = df.drop(columns=['FullyGrown'])
y = df['FullyGrown']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the XGBoost model
xg_reg = xgb.XGBClassifier(n_estimators=100, scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]))
xg_reg.set_params(early_stopping_rounds=5)
xg_reg.fit(X_train, y_train, eval_set=[(X_test, y_test)])

# Evaluate the model
y_pred = xg_reg.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mape = 100 * mae / y_test.mean()
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("Mean absolute error:", mae)
print("Mean absolute percentage error:", mape)
print("Root mean squared error:", rmse)
