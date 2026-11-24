
"""
SIMCHECK Phase 2
Uncertainty-based candidate ranking using
bootstrap model ensembles.
"""

import sys
import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


# Locate the Phase 2 directories and simulator

PHASE_ROOT = Path(__file__).resolve().parent.parent

DATASETS = PHASE_ROOT / "datasets"
MODELS_DIR = PHASE_ROOT / "models"
RESULTS = PHASE_ROOT / "results"

sys.path.insert(0, str(PHASE_ROOT / "01_simulation"))

from combat_env import VI_MAX_HP, JINX_MAX_HP


# 1. Define experiment parameters
# Preserve Phase 1's ensemble configuration

NUM_MEMBERS = 5
MAX_EPOCHS = 300

MODELS = ["random_forest", "neural_network"]

ENSEMBLES = MODELS_DIR / "ensembles"
ENSEMBLES.mkdir(exist_ok=True)

train = pd.read_csv(DATASETS / "train.csv")


# 2. Calculate predicted combat rewards
# Use the same reward function as candidate prediction

def calculate_reward(data, predictions):

    damage_dealt = (
        (data["enemy_hp"].to_numpy() - predictions[:, 1])
        / JINX_MAX_HP * 100
    )

    damage_received = (
        (data["player_hp"].to_numpy() - predictions[:, 0])
        / VI_MAX_HP * 100
    )

    reward = damage_dealt - damage_received

    reward += 100 * (
        predictions[:, 1] / JINX_MAX_HP * 100 < 0.5
    )

    reward -= 100 * (
        predictions[:, 0] / VI_MAX_HP * 100 < 0.5
    )

    return reward


# 3. Encode input variables

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


# 4. Train a separate ensemble for each architecture

for model_name in MODELS:

    print("\nMODEL:", model_name)

    bundle = joblib.load(
        MODELS_DIR / f"{model_name}.joblib"
    )

    features = bundle["features"]
    targets = bundle["targets"]

    candidates = pd.read_csv(
        RESULTS
        / f"{model_name}_candidate_predictions.csv"
    )

    X_candidates = encode(candidates, features)

    member_rewards = []


    # 5. Train five bootstrap models
    # Sample existing training rows with replacement
    # No additional simulator calls are required

    for member in range(NUM_MEMBERS):

        seed = 100 + member

        bootstrap = train.sample(
            n=len(train),
            replace=True,
            random_state=seed
        )

        X_train = encode(bootstrap, features)

        y_train = bootstrap[targets].to_numpy()


        # Train a bootstrap Random Forest

        if model_name == "random_forest":

            model = RandomForestRegressor(
                n_estimators=100,
                min_samples_leaf=3,
                random_state=seed,
                n_jobs=-1
            )

            model.fit(X_train, y_train)

            predictions = model.predict(X_candidates)

            member_bundle = {
                "model": model,
                "features": features,
                "targets": targets
            }


        # Train a bootstrap neural network

        else:

            input_scaler = StandardScaler()
            output_scaler = StandardScaler()

            X_scaled = input_scaler.fit_transform(X_train)

            y_scaled = output_scaler.fit_transform(y_train)

            model = MLPRegressor(
                hidden_layer_sizes=(64, 64),
                activation="relu",
                solver="adam",
                random_state=seed,
                early_stopping=False,
                max_iter=MAX_EPOCHS
            )

            for epoch in range(MAX_EPOCHS):

                model.partial_fit(X_scaled, y_scaled)

            predictions = model.predict(
                input_scaler.transform(X_candidates)
            )

            predictions = output_scaler.inverse_transform(
                predictions
            )

            member_bundle = {
                "model": model,
                "features": features,
                "targets": targets,
                "input_scaler": input_scaler,
                "output_scaler": output_scaler
            }


        # Confirm that predictions are valid

        assert np.isfinite(predictions).all()

        rewards = calculate_reward(
            candidates,
            predictions
        )

        assert np.isfinite(rewards).all()

        member_rewards.append(rewards)

        joblib.dump(
            member_bundle,
            ENSEMBLES / f"{model_name}_{member + 1}.joblib"
        )

        print(
            f"Bootstrap member {member + 1}"
            f"/{NUM_MEMBERS} complete."
        )


    # 6. Measure disagreement between ensemble members
    # Use sample standard deviation of predicted rewards

    reward_matrix = np.stack(member_rewards)

    candidates["action_uncertainty"] = np.std(
        reward_matrix,
        axis=0,
        ddof=1
    )


    # 7. Assign each state its maximum action uncertainty

    uncertainty = candidates.groupby(
        "state_id"
    )["action_uncertainty"].max()

    uncertainty = uncertainty.rename(
        "uncertainty_score"
    ).reset_index()


    # 8. Add uncertainty to our existing testing scores

    score_path = (
        RESULTS
        / f"{model_name}_testing_scores.csv"
    )

    scores = pd.read_csv(score_path)

    scores = scores.drop(
        columns=["uncertainty_score"],
        errors="ignore"
    )

    scores = scores.merge(
        uncertainty,
        on="state_id",
        validate="one_to_one"
    )

    assert len(scores) == 5000
    assert np.isfinite(scores["uncertainty_score"]).all()

    scores.to_csv(score_path, index=False)

    print("States ranked:", len(scores))

    print(
        "Maximum uncertainty:",
        round(scores["uncertainty_score"].max(), 6)
    )


print("\nGround-truth simulator queries: 0")
print("Uncertainty-based ranking complete.")
