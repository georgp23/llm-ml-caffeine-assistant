import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle
import os
import json

def train_personal_model(user_id):

    file_path="data/personal_ml_training_data.csv"
    model_output=f"models/personal_linear_regression_model_user_{user_id}.pkl"

    df = pd.read_csv(file_path)

    # Filter dataset by user_id
    df = df[df.iloc[:, -2] == user_id]  

    X = df.iloc[:, :-2]  # All columns except the last two (features)
    y = df.iloc[:, -1]   # The last column (feedback), target

    # Split data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    score = model.score(X_test, y_test)
    print(f"Model R^2 score for user_id {user_id}: {score:.3f}")

    # Save the trained model to a file
    with open(model_output, "wb") as f:
        pickle.dump(model, f)

    print(f"Model for user_id {user_id} saved successfully!")

    # Save the score to json file
    score_file = "data/user_models.json"
    if os.path.exists(score_file):
        with open(score_file, "r") as f:
            scores = json.load(f)
    else:
        scores = {}

    scores[str(user_id)] = {
        "score": round(score, 4),
        "coef": model.coef_.tolist(),
        "intercept": model.intercept_.tolist() if hasattr(model.intercept_, "tolist") else model.intercept_
    }


    with open(score_file, "w") as f:
        json.dump(scores, f, indent=2)

    return score
