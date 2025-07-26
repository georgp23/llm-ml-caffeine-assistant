import pickle
import numpy as np
import os

def predict_with_fed_global_model(input_df, model_path="models/federated_global_model.pkl"):


    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Global model not found at: {model_path}. Run aggregation first.")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Convert input to NumPy array
    X_input = np.array(input_df)
    predictions = model.predict(X_input)
    return predictions
