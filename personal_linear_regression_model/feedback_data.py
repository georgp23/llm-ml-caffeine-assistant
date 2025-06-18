import csv
import os
import pandas as pd

def feedback_logger(feedback, input_X, drink, expected):
    # One-hot encode the drink option
    drink_encoded = pd.get_dummies([drink], prefix="drink").iloc[0].to_dict()

    file_path = "personal_ml_training_data.csv"

    # Open the file in append mode
    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)

        # Construct the row from input_X + drink + effectiveness + feedback
        row = list(input_X.iloc[0])  # Assumes input_X is a single-row DataFrame
        row += list(drink_encoded.values())
        row += [expected, feedback]

        writer.writerow(row)
