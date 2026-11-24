
"""
SIMCHECK: Candidate State Ranking

Testing strategies:
1. Random testing
2. Action sensitivity
3. Risk-Aware Action Sensitivity (RAAS)

No ground-truth simulator queries are used.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Find the main Phase 1 folder
PHASE_ROOT = Path(__file__).resolve().parent.parent

# Find our existing results folder
RESULTS = PHASE_ROOT / "results"

# STEP 1: Define our world models

models = ["random_forest", "neural_network"]


# STEP 2: Calculate the testing scores
# separately for each world model

for name in models:

   # Load predictions from our existing results folder
    predictions = pd.read_csv(
        RESULTS / f"{name}_candidate_predictions.csv"
    )

    # Keep one row per starting combat state

    states = predictions.drop_duplicates(
        subset="state_id"
    ).copy()

    states = states[[
        "state_id",
        "player_hp",
        "enemy_hp",
        "distance",
        "enemy_attack",
        "dodge_cooldown"
    ]]

    states = states.sort_values("state_id")


    # METHOD 1: RANDOM TESTING
    #
    # Every state has an equal chance of selection
    # Use the same seed for both world models

    rng = np.random.default_rng(42)

    states["random_score"] = rng.random(len(states))


    # METHOD 2: ACTION SENSITIVITY
    #
    # Calculate the difference between the
    # highest and lowest predicted action rewards

    rewards = predictions.groupby(
        "state_id"
    )["predicted_reward"].agg(["max", "min"])

    rewards["reward_spread"] = (
        rewards["max"] - rewards["min"]
    )

    states = states.merge(
        rewards[["reward_spread"]],
        on="state_id"
    )

    states["sensitivity_score"] = (
        states["reward_spread"]
    )


    # METHOD 3: RISK-AWARE ACTION SENSITIVITY
    #
    # Combine three normalized risk factors


    # FACTOR A: Combat vulnerability
    #
    # Lower health means greater vulnerability
    # Consider whichever combatant is more vulnerable

    states["vulnerability"] = (
        1 - states[
            ["player_hp", "enemy_hp"]
        ].min(axis=1) / 100
    )


    # FACTOR B: Immediate danger
    #
    # Incoming attacks, proximity and unavailable
    # dodge abilities increase potential risk

    attack_present = (
        states["enemy_attack"] != "none"
    ).astype(int)

    proximity = (5 - states["distance"]) / 4

    dodge_unavailable = (
        states["dodge_cooldown"] > 0
    ).astype(int)

    states["danger"] = (
        attack_present
        * (1 + proximity + dodge_unavailable)
        / 3
    )


    # FACTOR C: Normalized action sensitivity
    #
    # Convert reward spread to a scale of 0 to 1

    minimum = states["reward_spread"].min()
    maximum = states["reward_spread"].max()

    if maximum > minimum:

        states["normalized_sensitivity"] = (
            (states["reward_spread"] - minimum)
            / (maximum - minimum)
        )

    else:

        states["normalized_sensitivity"] = 0.0


    # Combine our three factors with equal weights

    states["raas_score"] = (
        states["vulnerability"]
        + states["danger"]
        + states["normalized_sensitivity"]
    ) / 3


    # STEP 3: Save the testing scores in our existing results folder

    states.to_csv(
        RESULTS / f"{name}_testing_scores.csv",
        index=False
    )


    # STEP 4: Display the highest-ranked states

    print("\n" + "=" * 55)
    print(name.upper())
    print("=" * 55)

    methods = [
        "random_score",
        "sensitivity_score",
        "raas_score"
    ]

    for method in methods:

        ranked = states.sort_values(
            [method, "state_id"],
            ascending=[False, True]
        )

        print(f"\nTop 5 states: {method}")

        print(
            ranked[[
                "state_id",
                "player_hp",
                "enemy_hp",
                "distance",
                "enemy_attack",
                "dodge_cooldown",
                method
            ]].head(5).to_string(index=False)
        )


print("\nAll testing scores saved successfully.")
print("Ground-truth simulator queries used: 0")
