# Import the necessary libraries
import pandas as pd
import xgboost as xgb
import numpy as np
import json
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Read in the data
f = open("./data/measurements.json")
data_json = json.loads(f.read())["Measurements"]
vs = data_json.values()
df = pd.json_normalize(vs)

# Add an "ID" column by parsing it from the "RGB_Image" column
df["ID"] = df["RGB_Image"].str.strip(".png").str.strip("RGB_").apply(int)

# Encode the categorical variables as numerical values
le = LabelEncoder()
df['Variety'] = le.fit_transform(df['Variety'])
df['RGB_Image'] = le.fit_transform(df['RGB_Image'])
df['Depth_Information'] = le.fit_transform(df['Depth_Information'])

# Create the target column
df['FullyGrown'] = (df['Diameter'] >= 25).astype(int)

# Split the data into training and test sets
X = df.drop(columns=['FullyGrown'])
y = df['FullyGrown']

# Use k-fold cross-validation to evaluate the model
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Build the XGBoost model
    xg_reg = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1,
                              scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]))
    xg_reg.fit(X_train, y_train)

    # Evaluate the model
    y_pred = xg_reg.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = 100 * np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10)))
    scores.append([rmse, r2, mae, mape])

# Print the evaluation metrics
scores = np.array(scores)
print("Root mean squared error:", np.mean(scores[:,0]))
print("R^2 score:", np.mean(scores[:,1]))
print("Mean absolute error:", np.mean(scores[:,2]))
print("Mean absolute percentage error:", np.mean(scores[:,3]))
