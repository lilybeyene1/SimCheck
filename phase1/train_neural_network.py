
"""
SIMCHECK: Neural Network World Model

Purpose:
Train a second type of machine learning model
to predict combat outcomes.

Using multiple model architectures helps us test
whether our research findings apply beyond
Random Forests.
"""

import os
import joblib
import pandas as pd
import numpy as np


from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


# STEP 1: Load the same combat dataset.
data = pd.read_csv("combat_data.csv")


# STEP 2: Convert text categories into numbers.
data = pd.get_dummies(
    data,
    columns=["enemy_attack", "action"],
    dtype=int
)


# STEP 3: Define the outcomes we want to predict.
target_columns = [
    "next_player_hp",
    "next_enemy_hp",
    "next_distance",
    "next_dodge_cooldown"
]

X = data.drop(columns=target_columns)
y = data[target_columns]


# STEP 4: Use the same training and testing split
# as our Random Forest for a fair comparison.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# STEP 5: Create the neural network.
#
# Two hidden layers, each containing 64 neurons.
# ReLU helps the network learn nonlinear patterns.
# Adam adjusts the network's weights during training.

network = MLPRegressor(
    hidden_layer_sizes=(64, 64),
    activation="relu",
    solver="adam",
    max_iter=300,
    early_stopping=True,
    random_state=42
)


# STEP 6: Scale the input and output values.
#
# Health ranges up to 100, while distance only
# ranges from 1 to 5.
#
# Scaling prevents larger numerical values from
# unnecessarily dominating the learning process.

model = TransformedTargetRegressor(
    regressor=Pipeline([
        ("scaler", StandardScaler()),
        ("network", network)
    ]),
    transformer=StandardScaler()
)


# STEP 7: Train the neural network.
print("Training neural network world model...")

model.fit(X_train, y_train)

print("Training complete!")


# STEP 8: Evaluate the model on unseen examples.
predictions = model.predict(X_test)

# Check whether the model produced invalid numbers.
print("\nNumerical Stability Check:")
print("NaN values:", np.isnan(predictions).sum())
print("Infinite values:", np.isinf(predictions).sum())
print("All predictions finite:", np.isfinite(predictions).all())

# Check whether the neural network's learned weights are valid.
network = model.regressor_.named_steps["network"]

weights_valid = all(
    np.isfinite(weights).all()
    for weights in network.coefs_ + network.intercepts_
)

print("All model weights finite:", weights_valid)
print("Final training loss:", network.loss_)
print("Training iterations:", network.n_iter_)

# Calculate the average prediction errors.
print("\nTest Set Results:")

for i, column in enumerate(target_columns):

    error = mean_absolute_error(
        y_test[column],
        predictions[:, i]
    )

    print(f"{column}: MAE = {error:.3f}")


# STEP 9: Save the trained neural network.
# Keep it separate from our Random Forest model.

os.makedirs("models", exist_ok=True)

joblib.dump(
    {
        "model": model,
        "features": X.columns.tolist(),
        "targets": target_columns
    },
    "models/neural_network.joblib"
)

print("\nSaved model to models/neural_network.joblib")