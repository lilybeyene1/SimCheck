
"""
SIMCHECK: Uncertainty-Based Testing

Train bootstrap ensembles and measure disagreement
in predicted combat rewards.

I've selected five ensemble members, fixed training epochs 
and maximum reward disagreement as our uncertainty metric. 
These are experimental design choices, not established optimal settings. 
Ensemble disagreement is also an approximation of uncertainty, 
not a guarantee of prediction error.

No new ground-truth simulator queries are used.
"""

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


# STEP 1: Load existing training data.

train = pd.read_csv("datasets/train.csv")

targets = [
    "next_player_hp",
    "next_enemy_hp",
    "next_distance",
    "next_dodge_cooldown"
]

train = pd.get_dummies(
    train,
    columns=["enemy_attack", "action"],
    dtype=int
)

X_train = train.drop(columns=targets)
y_train = train[targets]

ENSEMBLE_SIZE = 5

os.makedirs("models/ensembles", exist_ok=True)


# STEP 2: Process each world model separately.

for name in ["random_forest", "neural_network"]:

    candidates = pd.read_csv(
        f"results/{name}_candidate_predictions.csv"
    )

    X_candidates = candidates[[
        "player_hp",
        "enemy_hp",
        "distance",
        "enemy_attack",
        "dodge_cooldown",
        "action"
    ]]

    X_candidates = pd.get_dummies(
        X_candidates,
        columns=["enemy_attack", "action"],
        dtype=int
    )

    X_candidates = X_candidates.reindex(
        columns=X_train.columns,
        fill_value=0
    )

    ensemble_rewards = []

    print("\n" + "=" * 55)
    print(name.upper())
    print("=" * 55)


    # STEP 3: Train five bootstrap models.

    for member in range(ENSEMBLE_SIZE):

        seed = 100 + member
        rng = np.random.default_rng(seed)

        # Sample training rows with replacement.

        indices = rng.choice(
            len(X_train),
            size=len(X_train),
            replace=True
        )

        X_boot = X_train.iloc[indices]
        y_boot = y_train.iloc[indices]


        if name == "random_forest":

            model = RandomForestRegressor(
                n_estimators=100,
                min_samples_leaf=3,
                random_state=seed,
                n_jobs=-1
            )

            model.fit(X_boot, y_boot)

            predictions = model.predict(X_candidates)

            saved = {"model": model}


        else:

            input_scaler = StandardScaler()
            output_scaler = StandardScaler()

            X_scaled = input_scaler.fit_transform(X_boot)
            y_scaled = output_scaler.fit_transform(y_boot)

            model = MLPRegressor(
                hidden_layer_sizes=(64, 64),
                activation="relu",
                solver="adam",
                random_state=seed,
                early_stopping=False
            )

            # Use the same fixed 300-epoch
            # training budget for every member.

            for epoch in range(300):

                model.partial_fit(X_scaled, y_scaled)

            candidate_scaled = input_scaler.transform(
                X_candidates
            )

            predictions = output_scaler.inverse_transform(
                model.predict(candidate_scaled)
            )

            saved = {
                "model": model,
                "input_scaler": input_scaler,
                "output_scaler": output_scaler
            }


        assert np.isfinite(predictions).all()

        saved["features"] = X_train.columns.tolist()
        saved["targets"] = targets

        joblib.dump(
            saved,
            f"models/ensembles/{name}_{member + 1}.joblib"
        )


        # STEP 4: Calculate predicted combat rewards
        # using our original reward function.

        damage_dealt = (
            candidates["enemy_hp"].to_numpy()
            - predictions[:, 1]
        )

        damage_taken = (
            candidates["player_hp"].to_numpy()
            - predictions[:, 0]
        )

        reward = (
            damage_dealt
            - damage_taken
            + 100 * (predictions[:, 1] < 0.5)
            - 100 * (predictions[:, 0] < 0.5)
        )

        ensemble_rewards.append(reward)

        print(
            f"Ensemble member {member + 1}"
            f"/{ENSEMBLE_SIZE} complete."
        )


    # STEP 5: Measure disagreement.
    #
    # Higher standard deviation means the ensemble
    # members disagree more about an action's reward.

    candidates["reward_uncertainty"] = np.std(
        np.array(ensemble_rewards),
        axis=0,
        ddof=1
    )


    # STEP 6: Give each state an uncertainty score.
    #
    # Prioritize the action with the greatest
    # uncertainty within each combat situation.

    uncertainty = candidates.groupby(
        "state_id"
    )["reward_uncertainty"].max()

    uncertainty = uncertainty.rename(
        "uncertainty_score"
    ).reset_index()


    # STEP 7: Add uncertainty to our existing scores.

    path = f"results/{name}_testing_scores.csv"

    scores = pd.read_csv(path)

    scores = scores.drop(
        columns=["uncertainty_score"],
        errors="ignore"
    )

    scores = scores.merge(
        uncertainty,
        on="state_id",
        validate="one_to_one"
    )

    scores.to_csv(path, index=False)


    print("\nTop 5 states by uncertainty:")

    print(
        scores.sort_values(
            ["uncertainty_score", "state_id"],
            ascending=[False, True]
        )[[
            "state_id",
            "player_hp",
            "enemy_hp",
            "distance",
            "enemy_attack",
            "dodge_cooldown",
            "uncertainty_score"
        ]].head(5).to_string(index=False)
    )


print("\nUncertainty scores saved successfully.")
print("Ground-truth simulator queries used: 0")


"""
Something interesting is emerging: both RAAS and uncertainty-based 
testing are prioritizing combat situations involving low health and 
incoming enemy attacks, despite using different scoring mechanisms.

That doesn't establish that either method is better, 
but it gives us an interesting comparison to investigate.
"""