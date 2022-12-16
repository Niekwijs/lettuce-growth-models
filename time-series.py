# Import required libraries
import os
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from cmath import sqrt

from xgboost import XGBRegressor

import exgee as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# Read the JSON data and normalize it into a Pandas dataframe
f = open("./data/measurements.json")
data_json = json.loads(f.read())["Measurements"]
vs = data_json.values()
df = pd.json_normalize(vs)

# Add an "ID" column by parsing it from the "RGB_Image" column
df["ID"] = df["RGB_Image"].str.strip(".png").str.strip("RGB_").apply(int)

# Sort the data by ID
df.sort_values("ID", inplace=True)

# Add a "Week" column using a custom function
prev = ""
counter = 0
def add_week_callback(item):
    global counter
    global prev
    if prev != item["Variety"] and item["Variety"] == "Satine":
        counter += 1
    prev = item.Variety
    return counter
df["Week"] = df.apply(func=add_week_callback, axis=1)


df.dtypes

"""
Variety,object
RGB_Image,object
Depth_Information,object
FreshWeightShoot,float64
DryWeightShoot,float64
Height,float64
Diameter,float64
LeafArea,float64
"""

# Split the data into train and test sets
train_data, test_data = train_test_split(df, test_size=0.2)

# Select only the columns with numerical data types
numeric_cols = [col for col in df.columns if df[col].dtype in [np.int64, np.float64]]

# Create a copy of the data with only the numerical columns
train_data = train_data[numeric_cols].copy()
test_data = test_data[numeric_cols].copy()

# Fill any missing values in the train and test data
# Note: we are no longer using the "subset" parameter here
train_data.fillna(train_data.mean(), inplace=True)
test_data.fillna(test_data.mean(), inplace=True)

# Train the xgboost model
model = XGBRegressor()
model.fit(train_data.drop(["ID", "Week"], axis=1), train_data["Diameter"])

# Use the trained model to make predictions for the test set
predictions = model.predict(test_data.drop(["ID", "Week"], axis=1))
predictions_df = pd.DataFrame(predictions)
df.rename(columns={"0":"Diameter"})




###

# Get the real values and the predicted values
real_values = test_data["Diameter"]
predicted_values = predictions

# Convert the values to pandas series
real_values = pd.Series(real_values)
predicted_values = pd.Series(predicted_values)

real_values = real_values.reset_index(drop=True)
predicted_values = predicted_values.reset_index(drop=True)

# Create a dataframe with the real and predicted values
results_df = pd.concat([real_values, predicted_values], axis=1)
results_df.rename(columns={0: "Predicted Diameter"}, inplace=True)
print(results_df)

# Evaluate the predictions against the true values
rmse = sqrt(mean_squared_error(real_values, predicted_values))
mae = mean_absolute_error(real_values, predicted_values)

# Calculate the percentage error
percentage_error = 100 * abs(real_values - predicted_values) / real_values
mape = np.mean(percentage_error)

# Print the results
print(f"Test RMSE: {rmse}")
print(f"Test MAE: {mae}")
print(f"mean absolute percentage error (MAPE) : {mape:.2f}%")

# Create a figure and axes
fig, ax = plt.subplots()

# Plot the real values in blue
ax.scatter(range(len(real_values)),real_values, c="b")

# Plot the predicted values in red
# Specify the x-coordinates and y-coordinates of the data points
ax.scatter(range(len(real_values)),predicted_values, c="r", marker="x")

# Add axis labels and a title
ax.set_xlabel("Data point index")
ax.set_ylabel("Diameter")
ax.set_title("Real vs predicted values")

# Add a legend to the plot
ax.legend(loc="upper right", labels=["Real values", "Predicted values"])

# Show the plot
plt.show()



# Create some fake data
data = {
    "Variety": ["Satine", "Satine", "Satine", "Satine", "Satine", "Other", "Other", "Other"],
    "FreshWeightShoot": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
    "DryWeightShoot": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "Height": [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5],
    "Diameter": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
}

# Convert the data to a pandas dataframe
fake_data = pd.DataFrame(data)

