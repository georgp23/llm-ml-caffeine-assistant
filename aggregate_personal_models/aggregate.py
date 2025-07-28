import os
import json
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

def aggregate_models():
    with open("data/user_models.json", "r") as f:
        user_models = json.load(f)

    deltas = []
    weights = []
    intercept_deltas = []

    for _, data in user_models.items():
        coef = np.array(data["coef"])
        intercept = data["intercept"]
        n_samples = data.get("n_samples", 20)  

        deltas.append(coef * n_samples)
        intercept_deltas.append(intercept * n_samples)
        weights.append(n_samples)

    total_samples = sum(weights)

    # Normalise aggregated updates
    avg_delta = np.sum(deltas, axis=0) / total_samples
    avg_intercept_delta = sum(intercept_deltas) / total_samples

    global_model = LinearRegression()

    # Initialse model by fitting dummy data to set attributes properly
    n_features = avg_delta.shape[0]
    X_dummy = np.zeros((1, n_features))
    y_dummy = np.zeros(1)
    global_model.fit(X_dummy, y_dummy)

    # overwrite coef and intercepts
    global_model.coef_ = avg_delta
    global_model.intercept_ = avg_intercept_delta

    save_path = "models/federated_global_model.pkl"

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(global_model, f)

    print(f"Federated global model saved to: {save_path}")



