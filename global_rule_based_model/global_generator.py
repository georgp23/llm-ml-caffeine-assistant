import random
import csv
from drink_profiles import drink_profiles
from rate_drink import rate_drink

# Define fixed categories
goal = ["energy", "focus", "relax", "sleep", "mood", "balance", "no_applicable_categories"]
time_of_day = ["morning", "afternoon", "evening", "night", "unknown"]
days_of_week = ["weekday", "weekend", "unknown"]
preferred_effects = ["calm_energy", "sustained_focus", "mood_boost", "energy", "clear_head", "gentle_stimulation", "no_applicable_categories"]
avoid_effects = ["jitters", "crash", "anxiety", "insomnia", "stomach_upset", "no_applicable_categories"]
urgency = ["low", "medium", "high", "no_applicable_categories"]
user_state = ["tired", "anxious", "wired", "foggy", "stressed", "rested", "no_applicable_categories"]

drinks = list(drink_profiles.keys())

with open("global_ml_training_data.csv", "w", newline="") as f:
    fieldnames = ["goal", "time_of_day", "user_state", "preferred_effects", "avoid_effects", "urgency", "drink", "effectiveness"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()  # Write the header once

    for num in range(2000):
        goal_choice = random.choice(goal)
        time_of_day_choice = random.choice(time_of_day)
        user_state_choice = random.choice(user_state)
        preferred_effect = random.choice(preferred_effects)
        avoid_effect = random.choice(avoid_effects)
        urgency_level = random.choice(urgency)
        drink = random.choice(drinks)

        # Pass the drink name and profiles to rate_drink
        effectiveness = rate_drink(
            goal_choice, time_of_day_choice, user_state_choice,
            [preferred_effect], [avoid_effect], urgency_level, drink, drink_profiles
        )

        writer.writerow({
            "goal": goal_choice,
            "time_of_day": time_of_day_choice,
            "user_state": user_state_choice,
            "preferred_effects": preferred_effect,
            "avoid_effects": avoid_effect,
            "urgency": urgency_level,
            "drink": drink,
            "effectiveness": effectiveness
        })

