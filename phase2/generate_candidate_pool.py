
"""
SIMCHECK Phase 2
Generate an unseen candidate testing pool.

No ground-truth simulator queries are used.
"""

import random
import pandas as pd

from pathlib import Path
from combat_env import (
    VI_MAX_HP,
    JINX_MAX_HP,
    VI_MAX_MANA,
    JINX_MAX_MANA,
    VI_Q_COOLDOWN,
    JINX_W_COOLDOWN,
    MAX_DISTANCE
)


# 1. Define experiment parameters.

ROOT = Path(__file__).resolve().parent

NUM_STATES = 5000

rng = random.Random(2026)

STATE_COLUMNS = [
    "player_hp",
    "enemy_hp",
    "distance",
    "player_mana",
    "enemy_mana",
    "player_cooldown",
    "enemy_cooldown",
    "enemy_attack"
]


# 2. Exclude all previously generated states.
# This includes training, validation and test data.

previous = pd.read_csv(ROOT / "combat_data.csv")

excluded = set(
    previous[STATE_COLUMNS].itertuples(
        index=False,
        name=None
    )
)

# Also exclude our manually inspected example.

excluded.add(
    (50, 200, 500, 295, 260, 0, 0, "ability")
)


# 3. Generate new candidate states.
# Use the same sampling distribution as the
# original Phase 2 training dataset.

records = []

seen = set()

while len(records) < NUM_STATES:

    state = (
        rng.randint(1, VI_MAX_HP),
        rng.randint(1, JINX_MAX_HP),

        round(rng.uniform(0, MAX_DISTANCE), 2),

        rng.randint(0, VI_MAX_MANA),
        rng.randint(0, JINX_MAX_MANA),

        0 if rng.random() < 0.5
        else rng.randint(1, VI_Q_COOLDOWN - 1),

        0 if rng.random() < 0.5
        else rng.randint(1, JINX_W_COOLDOWN - 1),

        rng.choice(["none", "attack", "ability"])
    )

    # Skip previously observed or duplicate states.

    if state in excluded or state in seen:
        continue

    seen.add(state)

    record = dict(zip(STATE_COLUMNS, state))

    record["state_id"] = len(records) + 1

    records.append(record)


# 4. Construct the candidate dataset.

pool = pd.DataFrame(records)

pool = pool[["state_id"] + STATE_COLUMNS]


# 5. Verify the candidate pool.

assert len(pool) == NUM_STATES

assert len(seen) == NUM_STATES

assert len(seen & excluded) == 0


# 6. Save the unlabeled candidate pool.

OUTPUT = ROOT / "datasets/candidate_pool.csv"

pool.to_csv(OUTPUT, index=False)


# 7. Display the results.

print("=" * 55)
print("SIMCHECK PHASE 2 - CANDIDATE POOL")
print("=" * 55)

print("Candidate states:", len(pool))
print("Unique states:", len(seen))
print("Previously observed overlap:", len(seen & excluded))

print("\nIncoming enemy actions:")

print(pool["enemy_attack"].value_counts())

print("\nGround-truth simulator queries: 0")

print("Saved to:", OUTPUT)

print("Candidate pool generation complete.")
