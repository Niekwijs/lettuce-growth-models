import os
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from cmath import sqrt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score, recall_score, precision_score, \
    accuracy_score
from sklearn.model_selection import train_test_split

def load_data(data_file):
    # Read the JSON data and normalize it into a Pandas dataframe
    with open(data_file) as f:
        data_json = json.loads(f.read())["Measurements"]
        vs = data_json.values()
        df = pd.json_normalize(vs)
    return df

def split_data(df):
    # Split the data into train and test sets
    # train_data, test_data = train_test_split(df, test_size=0.2)

    # Maak een TimeSeriesSplit-object met één split
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)

    # Splits de data in train- en testdata
    for train_index, test_index in tscv.split(df):
        train_data = df.iloc[train_index]
        test_data = df.iloc[test_index]

    # Select only the columns with numerical data types
    numeric_cols = [col for col in df.columns if df[col].dtype in [np.int64, np.float64]]

    # Create a copy of the data with only the numerical columns
    train_data = train_data[numeric_cols].copy()
    test_data = test_data[numeric_cols].copy()

    # Fill any missing values in the train and test data
    train_data.fillna(train_data.mean(), inplace=True)
    test_data.fillna(test_data.mean(), inplace=True)

    return train_data, test_data


def train_model(train_data):
    # Train the xgboost model
    model = XGBRegressor()
    model.fit(train_data, train_data["Diameter"])
    return model


def evaluate_model(model, test_data):
    # Use the trained model to make predictions for the test set
    predictions = model.predict(test_data.drop(["ID", "Week"], axis=1))
    predictions_df = pd.DataFrame(predictions)
    df.rename(columns={"0":"Diameter"})

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

    # Evaluate the predictions against the true values
    rmse = sqrt(mean_squared_error(real_values, predicted_values))
    mae = mean_absolute_error(real_values, predicted_values)

    # Calculate the percentage error
    percentage_error = 100 * abs(real_values - predicted_values) / real_values
    mape = np.mean(percentage_error)

    # Return the evaluation results
    return rmse, mae, mape, results_df, predictions


def predict_week(model, crop_diameter, train_data):
    # Predict the week in which the crop will reach the specified diameter
    weeks = model.predict(train_data)
    for i, diameter in enumerate(weeks):
        if diameter >= crop_diameter:
            return i
    return -1  # Return -1 if the crop never reaches the specified diameter

# def evaluate_model_week(model, test_data):
#     # Make predictions on the test set
#     predictions = model.predict(test_data)
#
#     # Extract the true labels from the test data
#     true_labels = test_data["Diameter"]
#
#     # Compute evaluation metrics
#     accuracy = accuracy_score(true_labels, predictions)
#     precision = precision_score(true_labels, predictions)
#     recall = recall_score(true_labels, predictions)
#     f1 = f1_score(true_labels, predictions)
#
#     # Print the evaluation metrics
#     print(f"Accuracy: {accuracy:.2f}")
#     print(f"Precision: {precision:.2f}")
#     print(f"Recall: {recall:.2f}")
#     print(f"F1 score: {f1:.2f}")
def evaluate_model_week(model, test_data, crop_diameter):
    # Predict the week in which the crop will reach the specified diameter
    weeks = model.predict(test_data.drop(["ID", "Week"], axis=1))
    predicted_weeks = []
    for i, diameter in enumerate(weeks):
        if diameter >= crop_diameter:
            predicted_weeks.append(i)

    # Extract the true weeks from the test data
    true_weeks = test_data["Week"]

    true_weeks = true_weeks[:len(predicted_weeks)]

    # Compute evaluation metrics
    accuracy = accuracy_score(true_weeks, predicted_weeks)
    # precision = precision_score(true_weeks, predicted_weeks)
    # recall = recall_score(true_weeks, predicted_weeks)
    # f1 = f1_score(true_weeks, predicted_weeks)

    # Print the evaluation metrics
    print(f"Accuracy: {accuracy:.2f}")
    # print(f"Precision: {precision:.2f}")
    # print(f"Recall: {recall:.2f}")
    # print(f"F1 score: {f1:.2f}")



def plot_results(results_df):
    # Get the real and predicted values from the results dataframe
    real_values = results_df["Diameter"]
    predicted_values = results_df["Predicted Diameter"]

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




if __name__ == "__main__":
    # Load the data
    df = load_data("./data/measurements.json")
    # Add an "ID" column by parsing it from the "RGB_Image" column
    df["ID"] = pd.Index(df["RGB_Image"]).str.extract("RGB_(\d+)", expand=False).astype(int)

    # Sort the data by ID
    df.sort_values("ID", inplace=True)

    # Add a "Week" column using a custom function
    df["Week"] = df.groupby("Variety").cumcount() + 1

    # Split the data
    train_data, test_data = split_data(df)

    # Train the model
    model = train_model(train_data.drop(["ID", "Week"], axis=1))


    test_dropped = test_data#.drop(["ID", "Week"], axis=1)

    evaluate_model_week(model, test_dropped, 20)



    ########################
    # Evaluate the model on the test data
    rmse, mae, mape, results_df, predictions = evaluate_model(model, test_data)


    # Print the evaluation results
    print(f"Test RMSE: {rmse:.2f}")
    print(f"Test MAE: {mae:.2f}")
    print(f"Test MAPE: {mape:.2f}%")

    plot_results(results_df)