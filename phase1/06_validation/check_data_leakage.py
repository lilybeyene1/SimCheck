
"""
SIMCHECK: Data Leakage Check

Purpose:
Determine whether our training and testing datasets
contain identical starting combat situations.

Overlapping states could make our models appear
better at generalizing than they actually are.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# Find the main Phase 1 folder
PHASE_ROOT = Path(__file__).resolve().parent.parent


# STEP 1: Load our original dataset

data = pd.read_csv(PHASE_ROOT / "combat_data.csv")


# STEP 2: Identify the variables that define
# a starting combat state
#
# We exclude the chosen action because the same
# starting state can have multiple possible actions

state_columns = [
    "player_hp",
    "enemy_hp",
    "distance",
    "enemy_attack",
    "dodge_cooldown"
]


# STEP 3: Recreate our original training/testing split
#
# Using the same random seed allows us to reproduce
# the split used when training both models

train, test = train_test_split(
    data,
    test_size=0.2,
    random_state=42
)


# STEP 4: Identify unique starting states

train_states = set(
    train[state_columns].itertuples(
        index=False, name=None
    )
)

test_states = set(
    test[state_columns].itertuples(
        index=False, name=None
    )
)


# STEP 5: Find states appearing in both datasets

overlapping_states = train_states.intersection(
    test_states
)

overlap_percentage = (
    len(overlapping_states) / len(test_states)
) * 100


# STEP 6: Check for identical state-action pairs
#
# This is a stricter form of overlap because both
# the starting situation AND the action are identical

input_columns = state_columns + ["action"]

train_inputs = set(
    train[input_columns].itertuples(
        index=False, name=None
    )
)

test_inputs = set(
    test[input_columns].itertuples(
        index=False, name=None
    )
)

overlapping_inputs = train_inputs.intersection(
    test_inputs
)


# STEP 7: Display our findings

print("=" * 50)
print("SIMCHECK: DATA LEAKAGE CHECK")
print("=" * 50)

print("\nDataset sizes:")
print("Total examples:", len(data))
print("Training examples:", len(train))
print("Testing examples:", len(test))

print("\nStarting state overlap:")
print("Unique training states:", len(train_states))
print("Unique testing states:", len(test_states))
print("Overlapping states:", len(overlapping_states))
print(f"State overlap percentage: {overlap_percentage:.2f}%")

print("\nIdentical state-action pairs:")
print("Unique training inputs:", len(train_inputs))
print("Unique testing inputs:", len(test_inputs))
print("Overlapping inputs:", len(overlapping_inputs))

print("\nExample overlapping states:")

for state in sorted(overlapping_states)[:5]:
    print(state)
