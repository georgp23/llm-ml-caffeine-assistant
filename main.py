import json
import pandas as pd
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
from global_rule_based_model.drink_profiles import drink_profiles
from LLM_interactions.parse_goal import parse_goal_to_json
from LLM_interactions.explain_drink_choice import explain_choices
from personal_linear_regression_model.feedback_data import feedback_logger

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

def main():
    print("Welcome to the Caffeine Recommendation System!")
    print("Describe your situation, and we'll recommend the best drink for you.")
    print("Type 'exit' to quit the program.")

    # Loop
    while True:
        # Get user input as a natural language prompt
        prompt = input("\nWhat do you need? (e.g., 'I need to focus tonight but I'm a bit anxious'): ").strip()

        # Exit the loop if the user types exit
        if prompt.lower() == "exit":
            print("Goodbye! Stay caffeinated responsibly!")
            break

        # Convert the user's input into structured JSON
        print("\nParsing your input into structured JSON...")
        parsed_json = parse_goal_to_json(prompt)

        # Convert parsed_json is a dictionary
        if isinstance(parsed_json, str):
            json_dict = json.loads(parsed_json)
        else:
            json_dict = parsed_json

        # Check if the parsing was successful
        if isinstance(json_dict, dict) and "error" in json_dict:
            print(f"Error: {json_dict['error']}")
            continue

        try:
            # Convert the parsed JSON string into a dictionary
            user_input = json.loads(parsed_json)

            # Preprocess the input for the model
            input_X = preprocess_input(user_input)

            # Get drink recommendations
            recommendations = recommend_drink(input_X)

            # Display the recommendations
            if "error" in recommendations:
                print(f"Error: {recommendations['error']}")
            else:
                # Extract the top recommendation (first dictionary in the list)
                top_recommendation = recommendations[0]  # Access the first element directly
                explanation = explain_choices([top_recommendation], prompt)
                print("\nExplanation for the recommendations:")
                print(explanation)

                # Take feedback from the suggestion
                feedback = int(input("\nRate this suggestion from 1-5: "))
                while feedback < 1 or feedback > 5:
                    print("Please provide a rating between 1 and 5.")
                    feedback = int(input("\nRate this suggestion from 1-5: "))

                # Pass json, drink, predicted effectiveness, and feedback to feedback_logger
                feedback_logger(
                    feedback,
                    input_X,
                    top_recommendation['drink'],  # Access the 'drink' key from the dictionary
                    top_recommendation['predicted_effectiveness']  # Access the 'predicted_effectiveness' key
                )
                
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
