
"""
SIMCHECK Phase 2
Freeze experimental rankings before accessing
ground-truth candidate outcomes.
"""

import hashlib
import numpy as np
import pandas as pd

from pathlib import Path


# 1. Define the experimental configuration
# Match Phase 1's budgets and repetitions

PHASE_ROOT = Path(__file__).resolve().parent.parent

DATASETS = PHASE_ROOT / "datasets"
RESULTS = PHASE_ROOT / "results"
EXPERIMENTS = PHASE_ROOT / "experiments"

EXPERIMENTS.mkdir(exist_ok=True)

MODELS = ["random_forest", "neural_network"]

QUERY_BUDGETS = [30, 75, 150, 300]

QUERIES_PER_STATE = 3

MAX_STATES = max(QUERY_BUDGETS) // QUERIES_PER_STATE

RANDOM_REPETITIONS = 100

METHODS = {
    "uncertainty": "uncertainty_score",
    "estimated_error": "estimated_error_score",
    "action_sensitivity": "sensitivity_score",
    "raas": "raas_score"
}


# 2. Load the candidate state IDs

pool = pd.read_csv(
    DATASETS / "candidate_pool.csv"
)

state_ids = np.sort(
    pool["state_id"].to_numpy()
)

records = []


# 3. Generate reproducible random rankings
# Use identical random selections for both models

random_rankings = {}

for repetition in range(RANDOM_REPETITIONS):

    rng = np.random.default_rng(
        1000 + repetition
    )

    random_rankings[repetition] = rng.choice(
        state_ids,
        size=MAX_STATES,
        replace=False
    )


# 4. Freeze rankings for each world model

for model_name in MODELS:

    scores = pd.read_csv(
        RESULTS
        / f"{model_name}_testing_scores.csv"
    )

    assert len(scores) == 5000

    assert scores["state_id"].is_unique

    assert set(scores["state_id"]) == set(state_ids)


    # Save all 100 random repetitions

    for repetition, ranking in random_rankings.items():

        for rank, state_id in enumerate(
            ranking,
            start=1
        ):

            records.append({
                "model": model_name,
                "method": "random",
                "repetition": repetition,
                "rank": rank,
                "state_id": state_id
            })


    # Rank the four nonrandom strategies
    # Break score ties using the state ID

    for method, score_column in METHODS.items():

        assert np.isfinite(scores[score_column]).all()

        ranking = scores.sort_values(
            [score_column, "state_id"],
            ascending=[False, True]
        ).head(MAX_STATES)

        for rank, state_id in enumerate(
            ranking["state_id"],
            start=1
        ):

            records.append({
                "model": model_name,
                "method": method,
                "repetition": 0,
                "rank": rank,
                "state_id": state_id
            })

    print("Rankings frozen:", model_name)


# 5. Construct and verify the experimental rankings

rankings = pd.DataFrame(records)

expected_rows = (
    len(MODELS)
    * (RANDOM_REPETITIONS + len(METHODS))
    * MAX_STATES
)

assert len(rankings) == expected_rows

assert not rankings.duplicated(
    subset=[
        "model",
        "method",
        "repetition",
        "state_id"
    ]
).any()


# 6. Save the frozen rankings

OUTPUT = EXPERIMENTS / "frozen_rankings.csv"

rankings.to_csv(OUTPUT, index=False)


# 7. Generate a SHA-256 fingerprint
# This lets us verify that the rankings remain
# unchanged during ground-truth evaluation

checksum = hashlib.sha256(
    OUTPUT.read_bytes()
).hexdigest()

(
    EXPERIMENTS / "rankings_sha256.txt"
).write_text(checksum)


# 8. Display the frozen experimental configuration

print("\n" + "=" * 55)
print("SIMCHECK PHASE 2 - FROZEN EXPERIMENT")
print("=" * 55)

print("World models:", len(MODELS))
print("Testing strategies:", len(METHODS) + 1)
print("Random repetitions:", RANDOM_REPETITIONS)
print("Query budgets:", QUERY_BUDGETS)
print("Maximum states per ranking:", MAX_STATES)
print("Frozen ranking entries:", len(rankings))

print("\nSHA-256 checksum:")
print(checksum)

print("\nGround-truth simulator queries: 0")
print("Experiment rankings frozen.")
