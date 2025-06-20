import pandas as pd
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
from global_rule_based_model.drink_profiles import drink_profiles

# Load the trained model and feature columns used during training
with open("drink_recommendation_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("feature_columns.pkl", "rb") as feature_file:
    feature_columns = pickle.load(feature_file)

# Preprocess user input to match the format the model expects
def preprocess_input(user_input):
    
    # Define all possible categories and effects (based on training data)
    categorical_cols = ["goal", "time_of_day", "user_state", "urgency"]
    all_effects = [
        "calm_energy", "sustained_focus", "mood_boost", "energy", "clear_head", "gentle_stimulation",
        "jitters", "crash", "anxiety", "insomnia", "stomach_upset", "no_applicable_categories"
    ]

    # Make sure all required keys are in the input, even if they're empty
    for col in categorical_cols:
        if col not in user_input:
            user_input[col] = "no_applicable_categories"

    # Handle missing or empty effects
    user_input["preferred_effects"] = user_input.get("preferred_effects", [])
    user_input["avoid_effects"] = user_input.get("avoid_effects", [])

    # If user_state is a list, convert it to a string (e.g. "anxious, tired")
    if isinstance(user_input.get("user_state"), list):
        user_input["user_state"] = ", ".join(user_input["user_state"])

    # Convert the input into a DataFrame (the format the model expects)
    input_df = pd.DataFrame([user_input])

    # One-hot encode the categorical column
    input_cat = pd.get_dummies(input_df[categorical_cols], prefix=categorical_cols)

    # Use mlb to encode the preferred and avoid effects
    mlb = MultiLabelBinarizer(classes=all_effects)
    preferred_effects = input_df["preferred_effects"].apply(lambda x: x if isinstance(x, list) else []).tolist()
    avoid_effects = input_df["avoid_effects"].apply(lambda x: x if isinstance(x, list) else []).tolist()

    input_preferred = pd.DataFrame(
        mlb.fit_transform(preferred_effects),
        columns=mlb.classes_,
        index=input_df.index
    ).add_prefix("pref_")

    input_avoid = pd.DataFrame(
        mlb.fit_transform(avoid_effects),
        columns=mlb.classes_,
        index=input_df.index
    ).add_prefix("avoid_")

    # Combine everything into one DataFrame
    input_X = pd.concat([input_cat, input_preferred, input_avoid], axis=1)

    # Reindex to match feature columns model was trained on
    input_X = input_X.reindex(columns=feature_columns, fill_value=0)

    return input_X

# Recommend the best drink based on the user input
def recommend_drink(input_X):
    try:
        results = []
        for drink in drink_profiles.keys():
            # Temporarily add the drink column to the input
            input_X[f"drink_{drink}"] = 1

            # Predict the effectiveness of drink
            predicted_effectiveness = model.predict(input_X)[0]

            # Save the result
            results.append({"drink": drink, "predicted_effectiveness": predicted_effectiveness})

            # Remove the drink column for the next iteration
            input_X[f"drink_{drink}"] = 0

        # Sort the results by predicted effectiveness (highest first)
        results = sorted(results, key=lambda x: x["predicted_effectiveness"], reverse=True)

        return results

    except Exception as e:
        return {"error": str(e)}