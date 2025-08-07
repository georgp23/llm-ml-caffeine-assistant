import os
import json
import pandas as pd
from LLM_interactions.parse_goal import parse_goal_to_json
from LLM_interactions.explain_drink_choice import explain_choices
from personal_linear_regression_model.feedback_data import feedback_logger
from personal_linear_regression_model.personal_model_train import train_personal_model
from personal_linear_regression_model.personal_predict import predict_with_personal_model
from global_rule_based_model.predict_drinks import preprocess_input, recommend_drink
from aggregate_personal_models.aggregate import aggregate_models
from aggregate_personal_models.predict_federated import predict_with_fed_global_model

def combine_predictions(global_score, personal_rating, personal_count, k=10):
    normalised_personal = (personal_rating - 3) / 2 # scale from [1, 5] to [-1, 1]
    alpha = 1 / (1 + personal_count / k) # confidence weighting
    final_score = alpha * global_score + (1 - alpha) * normalised_personal
    return round(final_score, 3)

def main():
    print("Welcome to the Caffeine Recommendation System!")
    print("Describe your situation, and we'll recommend the best drink for you.")
    print("Type 'exit' to quit the program.")

    # Loop
    while True:
        #Get user ID
        try:
            user_id = int(input("User ID: ").strip())
        except ValueError:
            print("Invalid User ID. Please enter a number.")
            continue
        
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
            user_input = json_dict

            # Preprocess the input for the model
            input_X = preprocess_input(user_input)

            try:
                # Use federated global model to score drinks
                drink_features_df = preprocess_input(user_input)  # DataFrame with one row per drink
                predicted_scores = predict_with_fed_global_model(drink_features_df)

                # Create recommendation list with drink names and scores
                drink_names = drink_features_df.index if drink_features_df.index.name == "drink" else list(drink_features_df.index)
                recommendations = [
                    {"drink": name, "predicted_effectiveness": score}
                    for name, score in zip(drink_names, predicted_scores)
                ]

                recommendations = sorted(recommendations, key=lambda x: x["predicted_effectiveness"], reverse=True)
                top_recommendation = recommendations[0]
                global_score = top_recommendation["predicted_effectiveness"]

                print("[Using federated global model for drink predictions]")

            except FileNotFoundError:
                print("[Federated model not found — falling back to rule-based global model]")
                recommendations = recommend_drink(input_X)

                if "error" in recommendations:
                    print(f"Error: {recommendations['error']}")
                    continue

                top_recommendation = recommendations[0]
                global_score = top_recommendation["predicted_effectiveness"]


            # Check for personal model
            try:
                personal_score = predict_with_personal_model(user_id, input_X)
                feedback_data = pd.read_csv("data/personal_ml_training_data.csv")
                personal_count = len(feedback_data[feedback_data.iloc[:, -1] == user_id])
                final_score = combine_predictions(global_score, personal_score, personal_count)

                explanation = explain_choices([top_recommendation], prompt)
                print("\nExplanation for the recommendations:")
                print(explanation)
                print(f"\nGlobal Score: {global_score:.2f}")
                print(f"Personal Model Score: {personal_score:.2f}")
                print(f"Combined Effectiveness Score: {final_score:.2f}")

            except FileNotFoundError:
                print("\n[Personal model not found — using global model only]")
                final_score = global_score
                personal_score = None

                explanation = explain_choices([top_recommendation], prompt)
                print("\nExplanation for the recommendations:")
                print(explanation)
                print(f"Predicted Effectiveness Score: {global_score:.2f}")

            # Get user feedback
            feedback = int(input("\nRate this suggestion from 1-5: "))
            while feedback < 1 or feedback > 5:
                print("Please provide a rating between 1 and 5.")
                feedback = int(input("\nRate this suggestion from 1-5: "))

            # Pass json, drink, predicted effectiveness, and feedback to feedback_logger
            feedback_logger(
                drink_features_df.loc[top_recommendation['drink']].to_dict(),
                top_recommendation['drink'],
                feedback,
                user_id
            )


                
        except Exception as e:
            print(f"An error occurred: {e}")

        # Try training personal model if enough records
        try:
            feedback_data = pd.read_csv("data/personal_ml_training_data.csv")
            record_count = len(feedback_data[feedback_data.iloc[:, -1] == user_id])
            # Update every 20 records
            if record_count >= 20 and record_count % 20 == 0:
                train_personal_model(user_id)
        except Exception as e:
            print(f"An error occurred: {e}")

        # Try federated global model if enough personal models exist
        try:
            model_dir = "models"
            prefix = "personal_linear_regression_model_user_"
            model_files = [f for f in os.listdir(model_dir) if f.startswith(prefix) and f.endswith(".pkl")]
            num_personal_models = len(model_files)
            if num_personal_models > 20:
                aggregate_models()

        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
