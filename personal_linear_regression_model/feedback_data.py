import csv
import os

def feedback_logger(feedback, input_X, drink, expected):
    with open("personal_ml_training_data.csv", "a", newline="") as f:
        # Add all columns from input_X plus drink, expected effectiveness, and feedback
        fieldnames = list(input_X.columns) + ["drink", "expected_effectiveness", "feedback"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        # Write the header if the file is new
        file_exists = os.path.isfile("personal_ml_training_data.csv")
        if not file_exists:
            writer.writeheader()

        # Add the drink, expected effectiveness, and feedback to the row
        row = input_X.iloc[0].to_dict()  # Convert the first row of input_X to a dictionary
        row.update({
            "drink": drink,
            "expected_effectiveness": expected,
            "feedback": feedback
        })

        writer.writerow(row)

