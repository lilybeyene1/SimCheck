
"""
SIMCHECK: Train Our First World Model

Purpose:
Train a machine learning model to predict combat outcomes.

The model learns from examples in combat_data.csv.
It does NOT directly access the combat rules in combat_env.py.

We will compare its predictions against the actual simulator
to identify mistakes, especially those that could lead an
AI agent to choose a bad action.

Model: Random Forest Regressor
"""

import os

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from pathlib import Path

# Find the main Phase 1 folder
PHASE_ROOT = Path(__file__).resolve().parent.parent

# Locations of our existing data, models, and results
DATASETS = PHASE_ROOT / "datasets"
MODELS = PHASE_ROOT / "models"
RESULTS = PHASE_ROOT / "results"

# STEP 1: Load our original combat dataset we generated earlier
data = pd.read_csv(PHASE_ROOT / "combat_data.csv")

print(f"Loaded {len(data)} combat examples.")


# STEP 2: Convert text categories into numeric columns
# Machine learning models need numerical inputs
#
# Example:
# action_attack = 1 means the player chose attack
# action_dodge = 1 means the player chose dodge
#
# This is called one-hot encoding
data = pd.get_dummies(
    data,
    columns=["enemy_attack", "action"],
    dtype=int
)


# STEP 3: Define what the model should predict
# These are the actual outcomes from our simulator
target_columns = [
    "next_player_hp",
    "next_enemy_hp",
    "next_distance",
    "next_dodge_cooldown"
]

# X contains the starting state and chosen action
# y contains the actual outcome
X = data.drop(columns=target_columns)
y = data[target_columns]


# STEP 4: Split the dataset into training and testing sets
#
# 80% is used to train the model
# 20% is held back to check its performance
#
# random_state=42 makes the split reproducible
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# STEP 5: Create the machine learning model
#
# n_estimators=100 means the forest contains 100 trees
# min_samples_leaf=3 helps reduce overfitting
# n_jobs=-1 allows scikit-learn to use available CPU cores
model = RandomForestRegressor(
    n_estimators=100,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)


# STEP 6: Train the model
# The forest learns relationships between states,
# actions, and outcomes from the training examples
print("\nTraining world model...")

model.fit(X_train, y_train)

print("Training complete!")


# STEP 7: Evaluate predictions on unseen examples
predictions = model.predict(X_test)

# MAE measures the average absolute prediction error
# Lower values indicate more accurate predictions
print("\nTest Set Results:")

for i, column in enumerate(target_columns):
    error = mean_absolute_error(
        y_test[column],
        predictions[:, i]
    )
    print(f"{column}: MAE = {error:.3f}")


# STEP 8: Save the trained model
# We will load it later when comparing testing methods
MODELS.mkdir(exist_ok=True) # Make sure our existing models folder is available

joblib.dump(
    {
        "model": model,
        "features": X.columns.tolist(),
        "targets": target_columns
    },
    MODELS / "random_forest.joblib"
)

print("\nSaved model to models/random_forest.joblib")


# STEP 9: Inspect one example prediction
#
# This is only a quick demonstration
# Our proper research evaluation will come later
example = X_test.iloc[[0]]
actual = y_test.iloc[0]
predicted = model.predict(example)[0]

print("\nExample prediction:")

for i, column in enumerate(target_columns):
    print(
        f"{column}: "
        f"actual={actual[column]}, "
        f"predicted={predicted[i]:.2f}"
    )
