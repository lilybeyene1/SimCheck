
"""
SIMCHECK: Estimated-Error-Based Testing

Train an auxiliary model to estimate world-model
reward prediction errors using existing validation data.

No new simulator queries are required.
"""

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor


# STEP 1: Load our existing validation data.

validation = pd.read_csv("datasets/validation.csv")

input_columns = [
    "player_hp",
    "enemy_hp",
    "distance",
    "enemy_attack",
    "dodge_cooldown",
    "action"
]


# STEP 2: Define our original combat reward function.

def calculate_rewards(data, next_player_hp, next_enemy_hp):

    damage_dealt = (
        data["enemy_hp"].to_numpy() - next_enemy_hp
    )

    damage_taken = (
        data["player_hp"].to_numpy() - next_player_hp
    )

    return (
        damage_dealt
        - damage_taken
        + 100 * (next_enemy_hp < 0.5)
        - 100 * (next_player_hp < 0.5)
    )


# Calculate actual rewards using existing labels.

actual_rewards = calculate_rewards(
    validation,
    validation["next_player_hp"].to_numpy(),
    validation["next_enemy_hp"].to_numpy()
)


# STEP 3: Encode the validation inputs.

encoded = pd.get_dummies(
    validation[input_columns],
    columns=["enemy_attack", "action"],
    dtype=int
)


# STEP 4: Process each world model separately.

for name in ["random_forest", "neural_network"]:

    saved = joblib.load(
        f"models/{name}_v2.joblib"
    )

    model = saved["model"]

    X_validation = encoded.reindex(
        columns=saved["features"],
        fill_value=0
    )


    # Generate world-model validation predictions.

    if name == "neural_network":

        X_scaled = saved["input_scaler"].transform(
            X_validation
        )

        predictions = saved[
            "output_scaler"
        ].inverse_transform(
            model.predict(X_scaled)
        )

    else:

        predictions = model.predict(X_validation)


    assert np.isfinite(predictions).all()


    predicted_rewards = calculate_rewards(
        validation,
        predictions[:, 0],
        predictions[:, 1]
    )


    # STEP 5: Measure actual prediction errors
    # on the existing validation dataset.

    absolute_errors = np.abs(
        actual_rewards - predicted_rewards
    )


    # STEP 6: Create features for our error estimator.

    X_error = X_validation.copy()

    X_error["predicted_reward"] = predicted_rewards

    X_error["predicted_player_hp"] = predictions[:, 0]

    X_error["predicted_enemy_hp"] = predictions[:, 1]


    # STEP 7: Train an auxiliary error prediction model.

    error_model = RandomForestRegressor(
        n_estimators=100,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

    error_model.fit(X_error, absolute_errors)

    joblib.dump(
        {
            "model": error_model,
            "features": X_error.columns.tolist()
        },
        f"models/{name}_error_estimator.joblib"
    )


    # STEP 8: Load the independent candidate predictions.
    #
    # These contain no actual simulator outcomes.

    candidates = pd.read_csv(
        f"results/{name}_candidate_predictions.csv"
    )

    X_candidates = pd.get_dummies(
        candidates[input_columns],
        columns=["enemy_attack", "action"],
        dtype=int
    )

    X_candidates["predicted_reward"] = (
        candidates["predicted_reward"]
    )

    X_candidates["predicted_player_hp"] = (
        candidates["predicted_next_player_hp"]
    )

    X_candidates["predicted_enemy_hp"] = (
        candidates["predicted_next_enemy_hp"]
    )

    X_candidates = X_candidates.reindex(
        columns=X_error.columns,
        fill_value=0
    )


    # STEP 9: Estimate candidate prediction errors.

    candidates["estimated_error"] = (
        error_model.predict(X_candidates)
    )

    assert np.isfinite(
        candidates["estimated_error"]
    ).all()


    # STEP 10: Assign each combat state a score.
    #
    # Use its largest estimated action reward error.

    error_scores = candidates.groupby(
        "state_id"
    )["estimated_error"].max()

    error_scores = error_scores.rename(
        "estimated_error_score"
    ).reset_index()


    # STEP 11: Add the final testing strategy.

    path = f"results/{name}_testing_scores.csv"

    scores = pd.read_csv(path)

    scores = scores.drop(
        columns=["estimated_error_score"],
        errors="ignore"
    )

    scores = scores.merge(
        error_scores,
        on="state_id",
        validate="one_to_one"
    )

    scores.to_csv(path, index=False)


    # STEP 12: Display our results.

    print("\n" + "=" * 55)
    print(name.upper())
    print("=" * 55)

    print("Error estimator training examples:", len(validation))

    print(
        "Mean validation reward error:",
        round(absolute_errors.mean(), 4)
    )

    print("\nTop 5 states by estimated error:")

    print(
        scores.sort_values(
            ["estimated_error_score", "state_id"],
            ascending=[False, True]
        )[[
            "state_id",
            "player_hp",
            "enemy_hp",
            "distance",
            "enemy_attack",
            "dodge_cooldown",
            "estimated_error_score"
        ]].head(5).to_string(index=False)
    )


print("\nEstimated-error scores saved successfully.")
print("Ground-truth simulator queries used: 0")



"""
Despite its relatively high prediction accuracy, 
our neural network still has situations where the 
auxiliary error estimator anticipates substantial mistakes.
"""