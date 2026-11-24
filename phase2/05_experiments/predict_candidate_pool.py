
"""
SIMCHECK Phase 2
Predict candidate combat outcomes using frozen
world models without querying the simulator.
"""

import sys
import numpy as np
import pandas as pd
import joblib

from pathlib import Path


# Locate the Phase 2 directories and simulator

PHASE_ROOT = Path(__file__).resolve().parent.parent

DATASETS = PHASE_ROOT / "datasets"
MODELS_DIR = PHASE_ROOT / "models"
RESULTS = PHASE_ROOT / "results"

sys.path.insert(0, str(PHASE_ROOT / "01_simulation"))

from combat_env import VI_MAX_HP, JINX_MAX_HP


# 1. Define experiment parameters

ACTIONS = ["attack", "ability", "retreat"]

MODELS = ["random_forest", "neural_network"]

KILL_REWARD = 100
DEATH_PENALTY = 100


# 2. Define the combat reward function
#
# Normalize health changes to percentages because
# the champions have different maximum health
#
# Preserve Phase 1's kill and death incentives

def calculate_reward(
    player_hp,
    enemy_hp,
    next_player_hp,
    next_enemy_hp
):

    damage_dealt = (
        (enemy_hp - next_enemy_hp)
        / JINX_MAX_HP * 100
    )

    damage_received = (
        (player_hp - next_player_hp)
        / VI_MAX_HP * 100
    )

    reward = damage_dealt - damage_received

    # Identify predicted deaths using a threshold
    # of 0.5 percentage points of maximum health

    reward += KILL_REWARD * (
        next_enemy_hp / JINX_MAX_HP * 100 < 0.5
    )

    reward -= DEATH_PENALTY * (
        next_player_hp / VI_MAX_HP * 100 < 0.5
    )

    return reward


# 3. Load the unlabeled candidate pool

pool = pd.read_csv(
    DATASETS / "candidate_pool.csv"
)


# 4. Generate all possible state-action combinations

candidate_actions = pd.concat(
    [
        pool.assign(action=action)
        for action in ACTIONS
    ],
    ignore_index=True
)

candidate_actions = candidate_actions.sort_values(
    "state_id",
    kind="stable"
).reset_index(drop=True)


# 5. Generate predictions for each model

for model_name in MODELS:

    bundle = joblib.load(
        MODELS_DIR / f"{model_name}.joblib"
    )

    model = bundle["model"]
    features = bundle["features"]
    targets = bundle["targets"]

    # Encode categorical input variables

    X = pd.get_dummies(
        candidate_actions.drop(columns=["state_id"]),
        columns=["enemy_attack", "action"],
        dtype=int
    )

    X = X.reindex(
        columns=features,
        fill_value=0
    )

    # Apply the saved neural network scalers

    if model_name == "neural_network":

        X_scaled = bundle[
            "input_scaler"
        ].transform(X)

        predictions = model.predict(X_scaled)

        predictions = bundle[
            "output_scaler"
        ].inverse_transform(predictions)

    else:

        predictions = model.predict(X)

    # Confirm that every prediction is finite

    assert np.isfinite(predictions).all()

    results = candidate_actions.copy()

    # Add predicted successor-state variables

    for i, target in enumerate(targets):

        results[f"predicted_{target}"] = (
            predictions[:, i]
        )

    # Calculate predicted utility for every action

    results["predicted_reward"] = calculate_reward(
        results["player_hp"],
        results["enemy_hp"],
        results["predicted_next_player_hp"],
        results["predicted_next_enemy_hp"]
    )

    assert np.isfinite(
        results["predicted_reward"]
    ).all()

    # Save the predictions for later testing

    results.to_csv(
        RESULTS
        / f"{model_name}_candidate_predictions.csv",
        index=False
    )

    print("\nMODEL:", model_name)

    print("Candidate states:", len(pool))
    print("Action predictions:", len(results))

    print(
        "Minimum predicted reward:",
        round(results["predicted_reward"].min(), 4)
    )

    print(
        "Maximum predicted reward:",
        round(results["predicted_reward"].max(), 4)
    )

    print("All predictions are finite.")


print("\nGround-truth simulator queries: 0")
print("Candidate prediction complete.")
