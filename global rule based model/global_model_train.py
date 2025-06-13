import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

# Load the dataset
df = pd.read_csv("global_ml_training_data.csv")

# Handle missing or empty values in preferred and avoid effects
df["preferred_effects"] = df["preferred_effects"].fillna("").apply(lambda x: x.split(",") if x else [])
df["avoid_effects"] = df["avoid_effects"].fillna("").apply(lambda x: x.split(",") if x else [])

# One-hot encode categorical columns
categorical_cols = ["goal", "time_of_day", "user_state", "urgency", "drink"]
df_cat = pd.get_dummies(df[categorical_cols], prefix=categorical_cols)

# Multi-label binarisation for preferred and avoid effects
mlb = MultiLabelBinarizer()

# Transform preferred effects
preferred = pd.DataFrame(
    mlb.fit_transform(df["preferred_effects"]),
    columns=mlb.classes_,
    index=df.index
).add_prefix("pref_")  # Add prefix for clarity

# Transform avoid effects
avoid = pd.DataFrame(
    mlb.fit_transform(df["avoid_effects"]),
    columns=mlb.classes_,
    index=df.index
).add_prefix("avoid_")  # Add prefix for clarity

# Combine all features into a single DataFrame
X = pd.concat([df_cat, preferred, avoid], axis=1)

# Target variable
y = df["effectiveness"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialise and train Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
score = model.score(X_test, y_test)
print("R^2 score:", score)
