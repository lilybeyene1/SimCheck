
"""
SIMCHECK: Revised World Model Training

Purpose:
Train both world models using state-disjoint datasets
and the revised numerical computing environment.

Original models remain unchanged.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from pathlib import Path

# Find the main Phase 1 folder
PHASE_ROOT = Path(__file__).resolve().parent.parent

# Locations of our existing data, models, and results
DATASETS = PHASE_ROOT / "datasets"
MODELS = PHASE_ROOT / "models"
RESULTS = PHASE_ROOT / "results"

# STEP 1: Load our new datasets
#
# We deliberately do not load the testing dataset, we load the training and validation datasets
# It is reserved for evaluation after model selection
train = pd.read_csv(DATASETS / "train.csv")
validation = pd.read_csv(DATASETS / "validation.csv")


# STEP 2: Identify what our models must predict
targets = [
    "next_player_hp",
    "next_enemy_hp",
    "next_distance",
    "next_dodge_cooldown"
]


# STEP 3: Convert categorical variables into numbers
def encode(dataset):

    return pd.get_dummies(
        dataset,
        columns=["enemy_attack", "action"],
        dtype=int
    )


train = encode(train)
validation = encode(validation)


# STEP 4: Separate inputs from expected outcomes
X_train = train.drop(columns=targets)
y_train = train[targets]

X_val = validation.drop(columns=targets)
y_val = validation[targets]

# Ensure both datasets have identical feature columns
X_val = X_val.reindex(
    columns=X_train.columns,
    fill_value=0
)


# STEP 5: Create output folders for our revised results if needed
MODELS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)


# STEP 6: Train the Random Forest
#
# Keep the original model settings for consistency
print("\nTRAINING RANDOM FOREST")

forest = RandomForestRegressor(
    n_estimators=100,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)

forest.fit(X_train, y_train)

print("Random Forest training complete!")


# STEP 7: Scale the neural network data

# Scaling helps the neural network learn variables
# that have different numerical ranges

# Scalers learn ONLY from the training dataset
input_scaler = StandardScaler()
output_scaler = StandardScaler()

X_train_scaled = input_scaler.fit_transform(X_train)
X_val_scaled = input_scaler.transform(X_val)

y_train_scaled = output_scaler.fit_transform(y_train)
y_val_scaled = output_scaler.transform(y_val)


# STEP 8: Create the neural network
#
# We keep the original two hidden layers
#
# We will manually monitor validation performance
# instead of using automatic early stopping
network = MLPRegressor(
    hidden_layer_sizes=(64, 64),
    activation="relu",
    solver="adam",
    random_state=42,
    early_stopping=False
)


# STEP 9: Prepare to monitor overfitting
training_losses = []
validation_losses = []

best_validation_loss = float("inf")
best_epoch = 0

patience = 20
epochs_without_improvement = 0

from copy import deepcopy

best_network = None
print("\nTRAINING NEURAL NETWORK")


# STEP 10: Train one epoch at a time
#
# An epoch means one pass through the training data
#
# After each epoch, we measure performance on
# both the training and validation datasets

for epoch in range(1, 301):

    network.partial_fit(
        X_train_scaled,
        y_train_scaled
    )

    train_predictions = network.predict(X_train_scaled)
    val_predictions = network.predict(X_val_scaled)

    # Calculate mean squared error on scaled outputs
    train_loss = np.mean(
        (y_train_scaled - train_predictions) ** 2
    )

    val_loss = np.mean(
        (y_val_scaled - val_predictions) ** 2
    )

    # Stop immediately if invalid numbers appear
    if not np.isfinite([train_loss, val_loss]).all():
        raise ValueError("Invalid numerical loss detected.")

    training_losses.append(train_loss)
    validation_losses.append(val_loss)

    # Preserve the best-performing network
    if val_loss < best_validation_loss:

        best_validation_loss = val_loss
        best_network = deepcopy(network)
        best_epoch = epoch

        epochs_without_improvement = 0

    else:

        epochs_without_improvement += 1

    if epoch % 10 == 0:

        print(
            f"Epoch {epoch}: "
            f"Training loss = {train_loss:.6f}, "
            f"Validation loss = {val_loss:.6f}"
        )

    # Stop if validation performance stops improving

    if epochs_without_improvement >= patience:

        print(f"Early stopping at epoch {epoch}")
        break


# Restore the network with the lowest validation error
network = best_network

print("\nNeural network training complete!")
print("Best epoch:", best_epoch)


# STEP 11: Generate predictions in original units
forest_train = forest.predict(X_train)
forest_val = forest.predict(X_val)

network_train = output_scaler.inverse_transform(
    network.predict(X_train_scaled)
)

network_val = output_scaler.inverse_transform(
    network.predict(X_val_scaled)
)


# STEP 12: Compare training and validation MAE
results = []

for name, train_pred, val_pred in [
    ("random_forest", forest_train, forest_val),
    ("neural_network", network_train, network_val)
]:

    print("\n" + "=" * 55)
    print(name.upper())
    print("=" * 55)

    assert np.isfinite(train_pred).all()
    assert np.isfinite(val_pred).all()

    for i, target in enumerate(targets):

        train_mae = mean_absolute_error(
            y_train[target], train_pred[:, i]
        )

        val_mae = mean_absolute_error(
            y_val[target], val_pred[:, i]
        )

        gap = val_mae - train_mae

        print(f"\n{target}")
        print(f"Training MAE:   {train_mae:.4f}")
        print(f"Validation MAE: {val_mae:.4f}")
        print(f"Error gap:      {gap:.4f}")

        results.append({
            "model": name,
            "target": target,
            "training_mae": train_mae,
            "validation_mae": val_mae,
            "error_gap": gap
        })


# STEP 13: Save the revised models separately
joblib.dump(
    {
        "model": forest,
        "features": X_train.columns.tolist(),
        "targets": targets
    },
    MODELS / "random_forest_v2.joblib"
)

joblib.dump(
    {
        "model": network,
        "input_scaler": input_scaler,
        "output_scaler": output_scaler,
        "features": X_train.columns.tolist(),
        "targets": targets,
        "best_epoch": best_epoch
    },
    MODELS / "neural_network_v2.joblib"
)


# STEP 14: Save our experimental results

pd.DataFrame(results).to_csv(
    RESULTS / "validation_results_v2.csv",
    index=False
)

pd.DataFrame({
    "epoch": range(1, len(training_losses) + 1),
    "training_loss": training_losses,
    "validation_loss": validation_losses
}).to_csv(
    RESULTS / "neural_network_learning_curve.csv",
    index=False
)


# STEP 15: Plot the neural network learning curves
plt.figure(figsize=(9, 5))

plt.plot(training_losses, label="Training loss")
plt.plot(validation_losses, label="Validation loss")

plt.axvline(
    best_epoch - 1,
    linestyle="--",
    label="Best epoch"
)

plt.xlabel("Epoch (zero-based)")
plt.ylabel("Mean squared error (scaled)")
plt.title("SIMCHECK: Neural Network Learning Curve")
plt.legend()
plt.tight_layout()

plt.savefig(
    RESULTS / "neural_network_learning_curve.png",
    dpi=300
)

print("\nAll predictions finite.")
print("Revised models saved successfully.")
print("Experimental results saved successfully.")

plt.show()
