# Import the necessary libraries
import pandas as pd
import xgboost as xgb
import numpy as np
import json

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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
df['FullyGrown'] = (df['Diameter'] >= 15).astype(int)

# Split the data into training and test sets
X = df.drop(columns=['FullyGrown'])
y = df['FullyGrown']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the XGBoost model
xg_reg = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
xg_reg.fit(X_train, y_train)

# Evaluate the model
y_pred = xg_reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mape = 100 * np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10)))

# Print the evaluation metrics
print("Root mean squared error:", rmse)
print("R^2 score:", r2)
print("Mean absolute error:", mae)
print("Mean absolute percentage error:", mape)

