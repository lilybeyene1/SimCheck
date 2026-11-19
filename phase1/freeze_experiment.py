
"""
SIMCHECK: Freeze Experimental Rankings

Save all testing priorities before revealing
any ground-truth candidate outcomes.

I've chosen 100 random repetitions to reduce the influence of 
lucky or unlucky random selections. However, these repetitions 
all use the same candidate pool, so they do not establish 
generalizability across independent populations. 
We'll need additional candidate pools for that.
"""

import os
import hashlib
import pandas as pd
import numpy as np

MODELS = ["random_forest", "neural_network"]

METHODS = {
    "random": "random_score",
    "uncertainty": "uncertainty_score",
    "estimated_error": "estimated_error_score",
    "action_sensitivity": "sensitivity_score",
    "raas": "raas_score"
}

BUDGETS = [30, 75, 150, 300]

RANDOM_REPETITIONS = 100

os.makedirs("experiments", exist_ok=True)

rankings = []

for model in MODELS:

    scores = pd.read_csv(
        f"results/{model}_testing_scores.csv"
    )

    # Verify that every candidate has valid scores.

    assert len(scores) == 5000
    assert scores["state_id"].is_unique

    for column in METHODS.values():

        assert np.isfinite(scores[column]).all()

    # Freeze the four nonrandom strategies.

    for method, column in METHODS.items():

        if method == "random":
            continue

        ranked = scores.sort_values(
            [column, "state_id"],
            ascending=[False, True]
        ).head(100)

        for rank, state_id in enumerate(
            ranked["state_id"], start=1
        ):

            rankings.append({
                "model": model,
                "method": method,
                "repetition": 0,
                "rank": rank,
                "state_id": state_id
            })

    # Repeat random testing 100 times.
    # Use the same seeds for both world models.

    state_ids = scores["state_id"].sort_values().to_numpy()

    for repetition in range(RANDOM_REPETITIONS):

        rng = np.random.default_rng(1000 + repetition)

        selected = rng.choice(
            state_ids,
            size=100,
            replace=False
        )

        for rank, state_id in enumerate(
            selected, start=1
        ):

            rankings.append({
                "model": model,
                "method": "random",
                "repetition": repetition,
                "rank": rank,
                "state_id": state_id
            })


# Save the frozen experimental rankings.

rankings = pd.DataFrame(rankings)

path = "experiments/frozen_rankings.csv"

rankings.to_csv(path, index=False)


# Generate a digital fingerprint of our rankings.
# This helps detect later modifications.

with open(path, "rb") as file:

    fingerprint = hashlib.sha256(
        file.read()
    ).hexdigest()

with open("experiments/rankings_sha256.txt", "w") as file:

    file.write(fingerprint)


print("SIMCHECK: EXPERIMENTAL DESIGN FROZEN")

print("\nWorld models:", len(MODELS))
print("Testing strategies:", len(METHODS))
print("Query budgets:", BUDGETS)
print("Random repetitions:", RANDOM_REPETITIONS)
print("Total ranking entries:", len(rankings))

print("\nSHA-256 fingerprint:")
print(fingerprint)

print("\nGround-truth simulator queries used: 0")
print("Rankings frozen successfully.")
