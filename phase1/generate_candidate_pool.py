
"""
SIMCHECK: Independent Candidate Pool

Purpose:
Generate 5,000 unique combat states for our main
budget-constrained testing experiment.

No simulator outcomes are generated yet.
"""

import random
import os
import pandas as pd


# STEP 1: Load our original dataset.

data = pd.read_csv("combat_data.csv")

state_columns = [
    "player_hp",
    "enemy_hp",
    "distance",
    "enemy_attack",
    "dodge_cooldown"
]


# STEP 2: Identify all previously used states.

excluded_states = set(
    data[state_columns].itertuples(
        index=False, name=None
    )
)


# Also exclude our four diagnostic scenarios.

excluded_states.update([
    (20, 40, 1, "skillshot", 0),
    (20, 40, 1, "skillshot", 2),
    (20, 10, 1, "basic", 0),
    (100, 100, 5, "none", 0)
])


# STEP 3: Configure our independent experiment.

rng = random.Random(2026)

POOL_SIZE = 5000

candidate_states = []
seen = set()


# STEP 4: Generate new combat situations.
#
# Reject duplicate states and any states used
# during our previous experiments.

while len(candidate_states) < POOL_SIZE:

    state = (
        rng.randint(5, 100),
        rng.randint(5, 100),
        rng.randint(1, 5),
        rng.choice(["none", "basic", "skillshot"]),
        rng.randint(0, 2)
    )

    if state in excluded_states or state in seen:
        continue

    seen.add(state)
    candidate_states.append(state)


# STEP 5: Convert our states into a dataset.

pool = pd.DataFrame(
    candidate_states,
    columns=state_columns
)

pool.insert(0, "state_id", range(1, len(pool) + 1))


# STEP 6: Verify independence and uniqueness.

overlap = len(seen & excluded_states)

assert overlap == 0
assert len(seen) == POOL_SIZE


# STEP 7: Save the candidate pool.

os.makedirs("datasets", exist_ok=True)

pool.to_csv(
    "datasets/candidate_pool.csv",
    index=False
)


# STEP 8: Display our results.

print("SIMCHECK: INDEPENDENT CANDIDATE POOL")

print("\nTotal candidate states:", len(pool))
print("Unique candidate states:", len(seen))
print("Previously used states:", overlap)
print("Simulator queries used:", 0)

print("\nEnemy attack distribution:")
print(pool["enemy_attack"].value_counts())

print("\nSample candidate states:")
print(pool.head())

print("\nCandidate pool saved successfully.")
