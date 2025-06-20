import json
import pandas as pd
from LLM_interactions.parse_goal import parse_goal_to_json
from LLM_interactions.explain_drink_choice import explain_choices
from personal_linear_regression_model.feedback_data import feedback_logger
from personal_linear_regression_model.personal_model_train import train_personal_model
from global_rule_based_model.predict_drinks import preprocess_input, recommend_drink

def main():
    print("Welcome to the Caffeine Recommendation System!")
    print("Describe your situation, and we'll recommend the best drink for you.")
    print("Type 'exit' to quit the program.")

    # Loop
    while True:
        #Get user ID
        user_id = input("User ID: ").strip()
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
                input_X.iloc[0].to_dict(),  # row as dict
                top_recommendation['drink'],
                feedback,
                user_id
            )

                
        except Exception as e:
            print(f"An error occurred: {e}")
        
        # Check how many records for user_id there are in personal_ml_training_data.csv
        try: 
            feedback_data = pd.read_csv("personal_ml_training_data.csv")
            record_count = len(feedback_data[feedback_data.iloc[:, -2] == user_id])
            
            # once enough records exist per user, train model
            if record_count > 20:
                train_personal_model(user_id)

        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
