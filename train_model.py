# train_model.py

import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load dataset
df = pd.read_csv("adult 3.csv")

# Required features and target
features = ["age", "education", "occupation", "gender", "hours-per-week"]
target = "income"

# Drop rows with missing values in required columns
df = df[features + [target]].dropna()

# Features and labels
X = df[features]
y = df[target]

# Preprocessing pipeline (categorical encoding)
categorical = ["education", "occupation", "gender"]
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical)],
    remainder="passthrough"
)

# Full pipeline
pipeline = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train the model
pipeline.fit(X, y)

# Save to income_model.pkl
with open("income_model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print(" Model trained and saved as 'income_model.pkl'")
