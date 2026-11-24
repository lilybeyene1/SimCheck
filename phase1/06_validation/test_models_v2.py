
"""
SIMCHECK: Final World Model Testing

Purpose:
Evaluate our revised models on the reserved test
dataset without retraining or changing their settings.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error


# Find the main Phase 1 folder
PHASE_ROOT = Path(__file__).resolve().parent.parent

# Point to our existing folders
DATASETS = PHASE_ROOT / "datasets"
MODELS = PHASE_ROOT / "models"
RESULTS = PHASE_ROOT / "results"


# STEP 1: Load the reserved testing dataset

test = pd.read_csv(DATASETS / "test.csv")


# STEP 2: Convert categorical variables into numbers

test = pd.get_dummies(
    test,
    columns=["enemy_attack", "action"],
    dtype=int
)


# STEP 3: Identify our prediction targets

targets = [
    "next_player_hp",
    "next_enemy_hp",
    "next_distance",
    "next_dodge_cooldown"
]


# STEP 4: Separate inputs from actual outcomes

X_test = test.drop(columns=targets)
y_test = test[targets]


# STEP 5: Load our recorded training
# and validation results

results = pd.read_csv(
    RESULTS / "validation_results_v2.csv"
)

results["testing_mae"] = np.nan


# STEP 6: Evaluate both revised models

models = ["random_forest", "neural_network"]

print("SIMCHECK: FINAL MODEL TESTING")
print("Testing examples:", len(test))

for name in models:

    saved = joblib.load(
        MODELS / f"{name}_v2.joblib"
    )

    model = saved["model"]

    # Ensure features match the training data

    X = X_test.reindex(
        columns=saved["features"],
        fill_value=0
    )

    # The neural network requires scaling
    # We reuse the scalers fitted during training
    # We NEVER fit new scalers on testing data

    if name == "neural_network":

        X_scaled = saved["input_scaler"].transform(X)

        predictions_scaled = model.predict(X_scaled)

        predictions = saved[
            "output_scaler"
        ].inverse_transform(predictions_scaled)

    else:

        predictions = model.predict(X)

    # Verify numerical stability

    assert np.isfinite(predictions).all()

    print("\n" + "=" * 55)
    print(name.upper())
    print("=" * 55)


    # STEP 7: Compare training, validation
    # and testing prediction errors

    for i, target in enumerate(targets):

        test_mae = mean_absolute_error(
            y_test[target],
            predictions[:, i]
        )

        mask = (
            (results["model"] == name)
            & (results["target"] == target)
        )

        train_mae = results.loc[
            mask, "training_mae"
        ].iloc[0]

        val_mae = results.loc[
            mask, "validation_mae"
        ].iloc[0]

        results.loc[mask, "testing_mae"] = test_mae

        print(f"\n{target}")
        print(f"Training MAE:   {train_mae:.4f}")
        print(f"Validation MAE: {val_mae:.4f}")
        print(f"Testing MAE:    {test_mae:.4f}")

        print(
            f"Train-test gap: "
            f"{test_mae - train_mae:.4f}"
        )


# STEP 8: Save the final evaluation results

results.to_csv(
    RESULTS / "final_model_results_v2.csv",
    index=False
)

print("\nAll predictions finite.")
print("Model testing complete!")
print("Results saved successfully.")
