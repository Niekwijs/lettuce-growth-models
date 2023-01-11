# Import the necessary libraries
import json

import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

f = open(r"./data/measurements.json")
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




# Create the target column
#!!! df['FullyGrown'] = df['FreshWeightShoot'] / df['Height']
#!!! df['FullyGrown'] = (df['Diameter'] >= 15).astype(int)
df['FullyGrown'] = df.Week

# Split the data into training and test sets
X = df.drop(columns=['FullyGrown',"Week","DryWeightShoot","FreshWeightShoot","ID","RGB_Image"])
y = df['FullyGrown']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the XGBoost model
xg_reg = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1)
xg_reg.fit(X_train, y_train)

# Evaluate the model
y_pred = xg_reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
# Calculate the absolute percentage error for each prediction
percentage_errors = np.abs((y_pred - y_test) / y_test)

# Calculate the mean absolute percentage error
mean_absolute_percentage_error = np.mean(percentage_errors) * 100

print(f'Mean absolute percentage error: {mean_absolute_percentage_error:.2f}%')
print(f'Root mean squared error: {rmse:.2f}')
print(f'Mean absolute error: {mae:.2f}')




import matplotlib.pyplot as plt

# Define the range of hyperparameters to evaluate
param_range = range(1, 200, 10)

# Initialize lists to store the training and test errors
train_errors = []
test_errors = []

# Iterate over the range of hyperparameters
for n in param_range:
    # Rebuild the model with the current number of trees
    xg_reg = xgb.XGBRegressor(n_estimators=n)

    # Fit the model on the training data
    xg_reg.fit(X_train, y_train)

    # Calculate the training and test errors
    y_train_pred = xg_reg.predict(X_train)
    y_test_pred = xg_reg.predict(X_test)
    train_error = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_error = np.sqrt(mean_squared_error(y_test, y_test_pred))

    # Append the errors to the lists
    train_errors.append(train_error)
    test_errors.append(test_error)

# Plot the training and test errors
plt.plot(param_range, train_errors, label='Training error')
plt.plot(param_range, test_errors, label='Test error')
plt.legend()
plt.xlabel('Number of trees')
plt.ylabel('RMSE')
plt.show()


# Plot the feature importance
xgb.plot_importance(xg_reg, importance_type='gain')
plt.show()