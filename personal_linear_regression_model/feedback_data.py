import csv
import os
from global_rule_based_model.drink_profiles import drink_profiles


def feedback_logger(input_X, drink, feedback, user_id):
    file_path = "data/personal_ml_training_data.csv"

    # One-hot encode the drink name
    drink_encoded = {f"drink_{d}": 1 if d == drink else 0 for d in drink_profiles.keys()}

    # Combine scenario + drink
    row = input_X.copy()
    for k, v in drink_encoded.items():
        row[k] = v

    # Add target and user ID
    row["feedback"] = feedback
    row["user_id"] = user_id

    # Consistent field order
    scenario_fields = [k for k in input_X.keys() if not k.startswith("drink_")]
    drink_fields = list(drink_encoded.keys())
    fieldnames = scenario_fields + drink_fields + ["feedback", "user_id"]

    file_exists = os.path.isfile(file_path)
    with open(file_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
