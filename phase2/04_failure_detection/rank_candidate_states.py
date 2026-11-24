
"""
SIMCHECK Phase 2
Rank candidate states using predicted action
sensitivity and risk-aware action sensitivity.
"""

import sys
import numpy as np
import pandas as pd

from pathlib import Path


# Locate the Phase 2 directories and simulator

PHASE_ROOT = Path(__file__).resolve().parent.parent

DATASETS = PHASE_ROOT / "datasets"
RESULTS = PHASE_ROOT / "results"

sys.path.insert(0, str(PHASE_ROOT / "01_simulation"))

from combat_env import (
    VI_MAX_HP,
    JINX_MAX_HP,
    MAX_DISTANCE,
    VI_Q_COST
)


# 1. Define experiment parameters

MODELS = ["random_forest", "neural_network"]

pool = pd.read_csv(
    DATASETS / "candidate_pool.csv"
)

pool = pool.sort_values("state_id")

rng = np.random.default_rng(42)

pool["random_score"] = rng.random(len(pool))


# 2. Calculate observable combat risk

# Vulnerability measures how close either
# champion is to losing all their health

pool["vulnerability"] = 1 - np.minimum(
    pool["player_hp"] / VI_MAX_HP,
    pool["enemy_hp"] / JINX_MAX_HP
)


# Identify whether an incoming attack exists

attack_present = (
    pool["enemy_attack"] != "none"
).astype(int)


# Closer champions receive higher proximity scores

proximity = 1 - pool["distance"] / MAX_DISTANCE


# Vi's ability is unavailable if she lacks
# sufficient mana or Q is on cooldown

ability_unavailable = (
    (pool["player_cooldown"] > 0)
    | (pool["player_mana"] < VI_Q_COST)
).astype(int)


# Combine observable indicators of danger

pool["danger"] = attack_present * (
    1 + proximity + ability_unavailable
) / 3


# 3. Calculate model-specific testing scores

for model_name in MODELS:

    predictions = pd.read_csv(
        RESULTS
        / f"{model_name}_candidate_predictions.csv"
    )

    # Action sensitivity measures the difference
    # between the best and worst predicted rewards

    spread = predictions.groupby(
        "state_id"
    )["predicted_reward"].agg(["min", "max"])

    spread["reward_spread"] = (
        spread["max"] - spread["min"]
    )

    scores = pool.merge(
        spread[["reward_spread"]],
        on="state_id",
        validate="one_to_one"
    )

    scores["sensitivity_score"] = (
        scores["reward_spread"]
    )


    # 4. Normalize sensitivity to a 0-1 scale

    minimum = scores["sensitivity_score"].min()
    maximum = scores["sensitivity_score"].max()

    if maximum > minimum:

        scores["normalized_sensitivity"] = (
            (scores["sensitivity_score"] - minimum)
            / (maximum - minimum)
        )

    else:

        scores["normalized_sensitivity"] = 0


    # 5. Calculate our proposed RAAS score
    # Preserve Phase 1's equal weighting

    scores["raas_score"] = (
        scores["normalized_sensitivity"]
        + scores["vulnerability"]
        + scores["danger"]
    ) / 3


    # 6. Verify and save the rankings

    assert len(scores) == 5000

    assert scores["raas_score"].between(0, 1).all()

    scores.to_csv(
        RESULTS
        / f"{model_name}_testing_scores.csv",
        index=False
    )

    print("\nMODEL:", model_name)

    print("States ranked:", len(scores))

    print(
        "Maximum action sensitivity:",
        round(maximum, 6)
    )

    print(
        "Maximum RAAS score:",
        round(scores["raas_score"].max(), 6)
    )


print("\nGround-truth simulator queries: 0")
print("Candidate ranking complete.")
