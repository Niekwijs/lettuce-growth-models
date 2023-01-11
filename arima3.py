import pandas as pd
import json
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

# Load the dataframe using the instructions provided
f = open("./data/measurements.json")
data_json = json.loads(f.read())["Measurements"]
vs = data_json.values()
df = pd.json_normalize(vs)

# Add the ID and Week columns
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

# Create a target column based on the diameter
df["FullyGrown"] = (df["Diameter"] >= 15).astype(int)

# Add a new column representing the time step for each measurement
df['TimeStep'] = range(len(df))

# Set the 'TimeStep' column as the index of the dataframe
df.set_index('TimeStep', inplace=True)

# Split the data into train and test sets using train_test_split
train_df, test_df = train_test_split(df, test_size=0.2)

# Fit the ARIMA model to the training data
arima_model = sm.tsa.ARIMA(train_df["Diameter"], order=(2,1,2)).fit()

# Use the model to make predictions on the test data, starting from the first time step
predictions = arima_model.predict(start=df['TimeStep'].min(), dynamic=True)

# Find the first week where the predicted diameter is 15 or more
for i, prediction in enumerate(predictions):
    if prediction >= 15:
        print(f"The plant is expected to be fully grown in week {i + 1}.")
        break
