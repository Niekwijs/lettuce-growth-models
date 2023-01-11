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

# Standardize the Diameter column
df["Diameter"] = (df["Diameter"] - df["Diameter"].mean()) / df["Diameter"].std()

# Split the data into train and test sets using train_test_split
train_df, test_df = train_test_split(df, test_size=0.2)

# Fit the ARIMA model to the training data
arima_model = sm.tsa.ARIMA(train_df["Diameter"], order=(2,1,2), exog=train_df[["Week", "LeafArea"]]).fit()

# Use the model to make predictions on the test data
predictions = arima_model.predict(start=test_df.index[0], end=test_df.index[-1], exog=test_df[["Week", "LeafArea"]], dynamic=True)

# Evaluate the model's performance
mse = ((predictions - test_df["Diameter"]) ** 2).mean()
mae = abs(predictions - test_df["Diameter"]).mean()
mape = (abs(predictions - test_df["Diameter"]) / test_df["Diameter"]).mean() * 100
print(f"Mean Squared Error: {mse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Absolute Percentage Error: {mape:.2f}%")
