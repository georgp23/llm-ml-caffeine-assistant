import csv
import os
import pandas as pd

def feedback_logger(feedback, input_X, drink, expected):
    # One-hot encode the drink option
    drink_encoded = pd.get_dummies([drink], prefix="drink").iloc[0].to_dict()

    with open("personal_ml_training_data.csv", "a", newline="") as f:
        # Add all columns from input_X plus one-hot encoded drink, expected effectiveness, and feedback
        fieldnames = list(input_X.columns) + list(drink_encoded.keys()) + ["expected_effectiveness", "feedback"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        # Write the header if the file is new
        file_exists = os.path.isfile("personal_ml_training_data.csv")
        if not file_exists:
            writer.writeheader()

        # Add the drink, expected effectiveness, and feedback to the row
        row = input_X.iloc[0].to_dict()  # Convert the first row of input_X to a dictionary
        row.update(drink_encoded)  # Add one-hot encoded drink values
        row.update({
            "expected_effectiveness": expected,
            "feedback": feedback
        })

        writer.writerow(row)

