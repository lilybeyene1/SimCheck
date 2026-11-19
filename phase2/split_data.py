
"""
SIMCHECK Phase 2
State-grouped training, validation and testing split.

Prevent identical starting combat states from
appearing in multiple datasets.
"""

import pandas as pd

from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit


# 1. Load the Phase 2 dataset.

ROOT = Path(__file__).resolve().parent

df = pd.read_csv(ROOT / "combat_data.csv")

OUTPUT = ROOT / "datasets"
OUTPUT.mkdir(exist_ok=True)


# 2. Identify all starting-state variables.
# Exclude the action and prediction targets.

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

groups = df.groupby(STATE_COLUMNS).ngroup()


# 3. Split into 70% training and 30% remaining.

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.30,
    random_state=42
)

train_idx, remaining_idx = next(
    splitter.split(df, groups=groups)
)

train = df.iloc[train_idx].copy()
remaining = df.iloc[remaining_idx].copy()


# 4. Divide the remaining data equally into
# validation and testing datasets.

remaining_groups = remaining.groupby(
    STATE_COLUMNS
).ngroup()

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.50,
    random_state=42
)

val_idx, test_idx = next(
    splitter.split(
        remaining,
        groups=remaining_groups
    )
)

validation = remaining.iloc[val_idx].copy()
test = remaining.iloc[test_idx].copy()


# 5. Confirm no starting-state overlap.

def get_states(data):

    return set(
        data[STATE_COLUMNS].itertuples(
            index=False,
            name=None
        )
    )


train_states = get_states(train)
val_states = get_states(validation)
test_states = get_states(test)

train_val_overlap = len(train_states & val_states)
train_test_overlap = len(train_states & test_states)
val_test_overlap = len(val_states & test_states)

assert train_val_overlap == 0
assert train_test_overlap == 0
assert val_test_overlap == 0

assert len(train) + len(validation) + len(test) == len(df)


# 6. Save the three datasets.

train.to_csv(OUTPUT / "train.csv", index=False)

validation.to_csv(
    OUTPUT / "validation.csv",
    index=False
)

test.to_csv(OUTPUT / "test.csv", index=False)


# 7. Display results.

print("=" * 55)
print("SIMCHECK PHASE 2 - DATA SPLITTING")
print("=" * 55)

for name, data in [
    ("Training", train),
    ("Validation", validation),
    ("Testing", test)
]:

    percentage = len(data) / len(df) * 100

    print(
        f"{name}: {len(data)} "
        f"({percentage:.2f}%)"
    )

print("\nStarting-state overlap:")

print("Train vs validation:", train_val_overlap)
print("Train vs test:", train_test_overlap)
print("Validation vs test:", val_test_overlap)

print("\nAdditional simulator queries: 0")
print("Data splitting complete.")
