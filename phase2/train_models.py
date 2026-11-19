
"""
SIMCHECK Phase 2
Train and evaluate both learned world models.

Reuse the Phase 1 model configurations in the
more complex League-inspired combat environment.
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# 1. Define file locations
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent

MODELS = ROOT / "models"
RESULTS = ROOT / "results"

MODELS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)


# --------------------------------------------------
# 2. Load the three datasets
# --------------------------------------------------

train = pd.read_csv(ROOT / "datasets/train.csv")
validation = pd.read_csv(ROOT / "datasets/validation.csv")
test = pd.read_csv(ROOT / "datasets/test.csv")


# --------------------------------------------------
# 3. Define inputs and prediction targets
# --------------------------------------------------

TARGETS = [
    "next_player_hp",
    "next_enemy_hp",
    "next_distance",
    "next_player_mana",
    "next_enemy_mana",
    "next_player_cooldown",
    "next_enemy_cooldown"
]


# Convert categorical inputs into numerical columns.

def encode(data):

    inputs = data.drop(columns=TARGETS)

    return pd.get_dummies(
        inputs,
        columns=["enemy_attack", "action"],
        dtype=int
    )


X_train = encode(train)

FEATURES = X_train.columns.tolist()

X_val = encode(validation).reindex(
    columns=FEATURES,
    fill_value=0
)

X_test = encode(test).reindex(
    columns=FEATURES,
    fill_value=0
)

y_train = train[TARGETS].to_numpy()
y_val = validation[TARGETS].to_numpy()
y_test = test[TARGETS].to_numpy()


# --------------------------------------------------
# 4. Train the Random Forest
# --------------------------------------------------

print("\nTRAINING RANDOM FOREST")

forest = RandomForestRegressor(
    n_estimators=100,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)

forest.fit(X_train, y_train)

joblib.dump(
    {
        "model": forest,
        "features": FEATURES,
        "targets": TARGETS
    },
    MODELS / "random_forest.joblib"
)

print("Random Forest training complete.")


# --------------------------------------------------
# 5. Scale neural network inputs and outputs
# --------------------------------------------------

# Fit scalers using ONLY the training dataset.
# This prevents information leakage.

input_scaler = StandardScaler()
output_scaler = StandardScaler()

X_train_scaled = input_scaler.fit_transform(X_train)
X_val_scaled = input_scaler.transform(X_val)
X_test_scaled = input_scaler.transform(X_test)

y_train_scaled = output_scaler.fit_transform(y_train)
y_val_scaled = output_scaler.transform(y_val)


# --------------------------------------------------
# 6. Configure the neural network
# --------------------------------------------------

print("\nTRAINING NEURAL NETWORK")

network = MLPRegressor(
    hidden_layer_sizes=(64, 64),
    activation="relu",
    solver="adam",
    random_state=42,
    early_stopping=False,
    max_iter=300
)

MAX_EPOCHS = 300
PATIENCE = 20

best_val_loss = float("inf")
best_epoch = 0

best_weights = None
best_biases = None

epochs_without_improvement = 0

history = []


# --------------------------------------------------
# 7. Train one epoch at a time
# --------------------------------------------------

for epoch in range(1, MAX_EPOCHS + 1):

    network.partial_fit(
        X_train_scaled,
        y_train_scaled
    )

    train_pred = network.predict(X_train_scaled)
    val_pred = network.predict(X_val_scaled)

    assert np.isfinite(train_pred).all()
    assert np.isfinite(val_pred).all()

    train_loss = np.mean(
        (y_train_scaled - train_pred) ** 2
    )

    val_loss = np.mean(
        (y_val_scaled - val_pred) ** 2
    )

    history.append({
        "epoch": epoch,
        "training_loss": train_loss,
        "validation_loss": val_loss
    })

    # Preserve the model with the lowest
    # validation error.

    if val_loss < best_val_loss:

        best_val_loss = val_loss
        best_epoch = epoch

        best_weights = [
            weight.copy()
            for weight in network.coefs_
        ]

        best_biases = [
            bias.copy()
            for bias in network.intercepts_
        ]

        epochs_without_improvement = 0

    else:

        epochs_without_improvement += 1

    if epoch % 25 == 0 or epoch == 1:

        print(
            f"Epoch {epoch:3d} | "
            f"Train: {train_loss:.6f} | "
            f"Validation: {val_loss:.6f}"
        )

    if epochs_without_improvement >= PATIENCE:

        print("Early stopping triggered.")
        break


# Restore the best-performing model.

network.coefs_ = best_weights
network.intercepts_ = best_biases

joblib.dump(
    {
        "model": network,
        "features": FEATURES,
        "targets": TARGETS,
        "input_scaler": input_scaler,
        "output_scaler": output_scaler,
        "best_epoch": best_epoch
    },
    MODELS / "neural_network.joblib"
)

print("\nBest epoch:", best_epoch)
print("Best validation loss:", best_val_loss)


# --------------------------------------------------
# 8. Save the neural network learning curve
# --------------------------------------------------

history_df = pd.DataFrame(history)

history_df.to_csv(
    RESULTS / "learning_curve.csv",
    index=False
)

plt.figure(figsize=(8, 5))

plt.plot(
    history_df["epoch"],
    history_df["training_loss"],
    label="Training"
)

plt.plot(
    history_df["epoch"],
    history_df["validation_loss"],
    label="Validation"
)

plt.axvline(
    best_epoch,
    linestyle="--",
    label="Best epoch"
)

plt.xlabel("Epoch")
plt.ylabel("Scaled mean squared error")
plt.title("Phase 2 Neural Network Learning Curve")
plt.legend()
plt.tight_layout()

plt.savefig(
    RESULTS / "learning_curve.png",
    dpi=200
)

plt.close()


# --------------------------------------------------
# 9. Generate predictions using both frozen models
# --------------------------------------------------

predictions = {
    "random_forest": [
        forest.predict(X_train),
        forest.predict(X_val),
        forest.predict(X_test)
    ],

    "neural_network": [
        output_scaler.inverse_transform(
            network.predict(X_train_scaled)
        ),
        output_scaler.inverse_transform(
            network.predict(X_val_scaled)
        ),
        output_scaler.inverse_transform(
            network.predict(X_test_scaled)
        )
    ]
}


# --------------------------------------------------
# 10. Calculate mean absolute error
# --------------------------------------------------

records = []

for model_name, outputs in predictions.items():

    for prediction in outputs:
        assert np.isfinite(prediction).all()

    train_mae = np.mean(
        np.abs(y_train - outputs[0]),
        axis=0
    )

    val_mae = np.mean(
        np.abs(y_val - outputs[1]),
        axis=0
    )

    test_mae = np.mean(
        np.abs(y_test - outputs[2]),
        axis=0
    )

    for i, target in enumerate(TARGETS):

        records.append({
            "model": model_name,
            "target": target,
            "training_mae": train_mae[i],
            "validation_mae": val_mae[i],
            "testing_mae": test_mae[i],
            "error_gap": test_mae[i] - train_mae[i]
        })


# --------------------------------------------------
# 11. Save and display the final results
# --------------------------------------------------

results = pd.DataFrame(records)

results.to_csv(
    RESULTS / "final_model_results.csv",
    index=False
)

print("\n" + "=" * 65)
print("SIMCHECK PHASE 2 - MODEL PERFORMANCE")
print("=" * 65)

print(
    results.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\nAll predictions are finite.")
print("Models trained: 2")
print("Additional simulator queries: 0")
print("Training and evaluation complete.")
