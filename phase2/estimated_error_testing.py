
"""
SIMCHECK Phase 2
Estimated prediction error testing.

Train an auxiliary model to predict reward errors
using the existing labeled validation dataset.
"""

import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

from combat_env import VI_MAX_HP, JINX_MAX_HP


# 1. Define experiment parameters.

ROOT = Path(__file__).resolve().parent

MODELS = ["random_forest", "neural_network"]

validation = pd.read_csv(
    ROOT / "datasets/validation.csv"
)


# 2. Calculate combat rewards.
#
# Ground-truth deaths occur at zero HP.
# Predicted deaths use our existing threshold
# of 0.5 percentage points of maximum health.

def calculate_reward(
    data,
    next_player_hp,
    next_enemy_hp,
    ground_truth=False
):

    damage_dealt = (
        (data["enemy_hp"].to_numpy() - next_enemy_hp)
        / JINX_MAX_HP * 100
    )

    damage_received = (
        (data["player_hp"].to_numpy() - next_player_hp)
        / VI_MAX_HP * 100
    )

    reward = damage_dealt - damage_received

    if ground_truth:

        reward += 100 * (next_enemy_hp <= 0)
        reward -= 100 * (next_player_hp <= 0)

    else:

        reward += 100 * (
            next_enemy_hp / JINX_MAX_HP * 100 < 0.5
        )

        reward -= 100 * (
            next_player_hp / VI_MAX_HP * 100 < 0.5
        )

    return reward


# 3. Encode categorical input variables.

def encode(data, features):

    encoded = pd.get_dummies(
        data,
        columns=["enemy_attack", "action"],
        dtype=int
    )

    return encoded.reindex(
        columns=features,
        fill_value=0
    )


# 4. Train one error estimator per world model.

for model_name in MODELS:

    print("\nMODEL:", model_name)

    bundle = joblib.load(
        ROOT / f"models/{model_name}.joblib"
    )

    model = bundle["model"]
    features = bundle["features"]

    X_val = encode(validation, features)


    # 5. Generate validation predictions.

    if model_name == "neural_network":

        predictions = model.predict(
            bundle["input_scaler"].transform(X_val)
        )

        predictions = bundle[
            "output_scaler"
        ].inverse_transform(predictions)

    else:

        predictions = model.predict(X_val)


    # 6. Calculate actual and predicted rewards.
    # Actual outcomes come from existing labels,
    # not new simulator queries.

    actual_rewards = calculate_reward(
        validation,
        validation["next_player_hp"].to_numpy(),
        validation["next_enemy_hp"].to_numpy(),
        ground_truth=True
    )

    predicted_rewards = calculate_reward(
        validation,
        predictions[:, 0],
        predictions[:, 1]
    )

    absolute_errors = np.abs(
        actual_rewards - predicted_rewards
    )


    # 7. Construct the error estimator inputs.
    # Use observable states and model predictions.
    # Do not include ground-truth outcomes.

    X_error = X_val.copy()

    X_error["predicted_reward"] = predicted_rewards

    X_error["predicted_next_player_hp"] = (
        predictions[:, 0]
    )

    X_error["predicted_next_enemy_hp"] = (
        predictions[:, 1]
    )


    # 8. Train the auxiliary error estimator.
    # Preserve Phase 1's model configuration.

    estimator = RandomForestRegressor(
        n_estimators=100,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

    estimator.fit(X_error, absolute_errors)

    joblib.dump(
        {
            "model": estimator,
            "features": X_error.columns.tolist()
        },
        ROOT / "models"
        / f"{model_name}_error_estimator.joblib"
    )


    # 9. Load unseen candidate predictions.

    candidates = pd.read_csv(
        ROOT / "results"
        / f"{model_name}_candidate_predictions.csv"
    )

    X_candidates = encode(candidates, features)

    for column in [
        "predicted_reward",
        "predicted_next_player_hp",
        "predicted_next_enemy_hp"
    ]:

        X_candidates[column] = candidates[column]


    # 10. Estimate errors without ground truth.

    candidates["estimated_action_error"] = (
        estimator.predict(X_candidates)
    )

    assert np.isfinite(
        candidates["estimated_action_error"]
    ).all()


    # 11. Assign each state its maximum
    # estimated error across available actions.

    estimated_errors = candidates.groupby(
        "state_id"
    )["estimated_action_error"].max()

    estimated_errors = estimated_errors.rename(
        "estimated_error_score"
    ).reset_index()


    # 12. Add the final testing strategy
    # to our existing candidate scores.

    score_path = (
        ROOT / "results"
        / f"{model_name}_testing_scores.csv"
    )

    scores = pd.read_csv(score_path)

    scores = scores.drop(
        columns=["estimated_error_score"],
        errors="ignore"
    )

    scores = scores.merge(
        estimated_errors,
        on="state_id",
        validate="one_to_one"
    )

    assert len(scores) == 5000

    scores.to_csv(score_path, index=False)


    # 13. Display results.

    print("Validation transitions:", len(validation))

    print(
        "Validation reward MAE:",
        round(absolute_errors.mean(), 6)
    )

    print("States ranked:", len(scores))

    print(
        "Maximum estimated error:",
        round(scores["estimated_error_score"].max(), 6)
    )


print("\nGround-truth simulator queries: 0")
print("Estimated error ranking complete.")
