import csv
import os

def feedback_logger(feedback, scenario, drink, expected):

    with open("personal_ml_training_data.csv", "a", newline="") as f:
        fieldnames = ["goal", "time_of_day", "user_state", "day_of_week", "preferred_effects", "avoid_effects", "urgency", "drink", "effectiveness" "feedback"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        file_exists = os.path.isfile("global_ml_training_data.csv")
        if not file_exists:
            writer.writeheader()  # Write the header only if the file does not exist

        goal = scenario["goal"]
        time_of_day = scenario["time_of_day"]
        user_state = scenario["goal"]
        day_of_week = scenario["day_of_week"]
        preferred_effect = scenario["preferred_effects"] # Should be one hot encoded as are lists
        avoid_effect = scenario["avoid_effects"] 
        urgency_level = scenario["urgency"]
        drink = drink
        expected_effectiveness = expected
        feedback = feedback

        writer.writerow({
            "goal": goal,
            "time_of_day": time_of_day,
            "user_state": user_state,
            "day_of_week": day_of_week,
            "preferred_effects": preferred_effect,
            "avoid_effects": avoid_effect,
            "urgency": urgency_level,
            "drink": drink,
            "expected_effectiveness": expected_effectiveness,
            "feedback": feedback
        })

