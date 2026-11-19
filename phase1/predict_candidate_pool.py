
"""
SIMCHECK: Candidate Pool Predictions

Purpose:
Generate predicted outcomes for every possible
action in our independent candidate pool.

No ground-truth simulator queries are used.
"""

import os
import joblib
import numpy as np
import pandas as pd


# STEP 1: Load our independent candidate pool.

pool = pd.read_csv("datasets/candidate_pool.csv")

ACTIONS = ["attack", "dodge", "retreat"]


# STEP 2: Assign every possible action
# to each starting combat state.

action_data = []

for action in ACTIONS:

    states = pool.copy()
    states["action"] = action

    action_data.append(states)

candidates = pd.concat(
    action_data,
    ignore_index=True
)

candidates = candidates.sort_values(
    "state_id"
).reset_index(drop=True)


# STEP 3: Encode categorical variables.

encoded = pd.get_dummies(
    candidates.drop(columns=["state_id"]),
    columns=["enemy_attack", "action"],
    dtype=int
)


# STEP 4: Identify our revised world models.

models = ["random_forest", "neural_network"]

os.makedirs("results", exist_ok=True)

print("SIMCHECK: CANDIDATE POOL PREDICTIONS")
print("Unique starting states:", len(pool))
print("State-action combinations:", len(candidates))


# STEP 5: Generate predictions from each model.

for name in models:

    saved = joblib.load(
        f"models/{name}_v2.joblib"
    )

    model = saved["model"]

    X = encoded.reindex(
        columns=saved["features"],
        fill_value=0
    )

    # Apply the neural network's original scalers.

    if name == "neural_network":

        X_scaled = saved["input_scaler"].transform(X)

        scaled_predictions = model.predict(X_scaled)

        predictions = saved[
            "output_scaler"
        ].inverse_transform(scaled_predictions)

    else:

        predictions = model.predict(X)


    # STEP 6: Verify that predictions are valid numbers.

    assert np.isfinite(predictions).all()

    results = candidates.copy()

    for i, target in enumerate(saved["targets"]):

        results[f"predicted_{target}"] = predictions[:, i]


    # STEP 7: Calculate predicted combat rewards.
    #
    # Use the same reward function as our earlier
    # diagnostic experiments.
    #
    # Damage dealt: +1 per HP
    # Damage taken: -1 per HP
    # Enemy killed: +100
    # Player killed: -100

    damage_dealt = (
        results["enemy_hp"]
        - results["predicted_next_enemy_hp"]
    )

    damage_taken = (
        results["player_hp"]
        - results["predicted_next_player_hp"]
    )

    enemy_killed = (
        results["predicted_next_enemy_hp"] < 0.5
    )

    player_killed = (
        results["predicted_next_player_hp"] < 0.5
    )

    results["predicted_reward"] = (
        damage_dealt
        - damage_taken
        + 100 * enemy_killed
        - 100 * player_killed
    )


    # STEP 8: Save predictions for later experiments.

    results.to_csv(
        f"results/{name}_candidate_predictions.csv",
        index=False
    )

    print("\n" + "=" * 55)
    print(name.upper())
    print("=" * 55)

    print("Total predictions:", len(results))
    print("All predictions finite: True")

    print("\nSample predicted rewards:")

    print(
        results[
            ["state_id", "action", "predicted_reward"]
        ].head(9).to_string(index=False)
    )

    print("\nPredictions saved successfully.")


print("\nTotal model predictions:", len(candidates) * len(models))
print("Ground-truth simulator queries used: 0")
