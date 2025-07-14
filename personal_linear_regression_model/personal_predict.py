import pickle
import numpy as np
import os

def predict_with_personal_model(user_id, input_features):

    model_path = f"models/personal_linear_regression_model_user_{user_id}.pkl"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found for user_id {user_id}. Train the model first.")
    
    # Load the model
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    input_array = np.array(input_features).reshape(1, -1)
    
    prediction = model.predict(input_array)[0]
    return prediction
