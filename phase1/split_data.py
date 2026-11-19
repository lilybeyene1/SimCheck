
"""
SIMCHECK: Leakage-Free Dataset Splitting

Purpose:
Create independent training, validation, and testing
datasets without overlapping starting combat states.

Training:   70%
Validation: 15%
Testing:    15%

The original dataset remains unchanged.
"""

import os
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


# STEP 1: Load the original dataset.

data = pd.read_csv("combat_data.csv")


# STEP 2: Identify the starting combat state.
#
# We exclude the action and predicted outcomes
# because they do not define the starting situation.

state_columns = [
    "player_hp",
    "enemy_hp",
    "distance",
    "enemy_attack",
    "dodge_cooldown"
]


# STEP 3: Assign a group ID to each unique state.
#
# Rows with identical starting states receive
# the same group ID.

groups = data.groupby(state_columns).ngroup()


# STEP 4: Separate training data from the remaining data.
#
# GroupShuffleSplit keeps identical starting states
# together rather than splitting individual rows.

first_split = GroupShuffleSplit(
    n_splits=1,
    test_size=0.30,
    random_state=42
)

train_indices, remaining_indices = next(
    first_split.split(data, groups=groups)
)

train = data.iloc[train_indices].copy()
remaining = data.iloc[remaining_indices].copy()


# STEP 5: Divide the remaining 30% equally
# into validation and testing datasets.

remaining_groups = groups.iloc[remaining_indices]

second_split = GroupShuffleSplit(
    n_splits=1,
    test_size=0.50,
    random_state=42
)

validation_indices, test_indices = next(
    second_split.split(
        remaining,
        groups=remaining_groups
    )
)

validation = remaining.iloc[validation_indices].copy()
test = remaining.iloc[test_indices].copy()


# STEP 6: Extract unique starting states
# from each dataset.

def get_states(dataset):

    return set(
        dataset[state_columns].itertuples(
            index=False,
            name=None
        )
    )


train_states = get_states(train)
validation_states = get_states(validation)
test_states = get_states(test)


# STEP 7: Verify that no starting states overlap.

train_val_overlap = len(
    train_states & validation_states
)

train_test_overlap = len(
    train_states & test_states
)

val_test_overlap = len(
    validation_states & test_states
)


# Stop the program if any overlap is detected.

assert train_val_overlap == 0
assert train_test_overlap == 0
assert val_test_overlap == 0

# Confirm that no original examples were lost.

assert len(train) + len(validation) + len(test) == len(data)


# STEP 8: Save the new datasets.
#
# A separate folder preserves our original dataset.

os.makedirs("datasets", exist_ok=True)

train.to_csv("datasets/train.csv", index=False)
validation.to_csv("datasets/validation.csv", index=False)
test.to_csv("datasets/test.csv", index=False)


# STEP 9: Display the results.

print("=" * 55)
print("SIMCHECK: LEAKAGE-FREE DATASET SPLIT")
print("=" * 55)

print("\nDataset sizes:")

for name, dataset in [
    ("Training", train),
    ("Validation", validation),
    ("Testing", test)
]:

    percentage = len(dataset) / len(data) * 100

    print(
        f"{name}: {len(dataset)} examples "
        f"({percentage:.2f}%)"
    )

print("\nStarting state overlap:")

print("Training vs. Validation:", train_val_overlap)
print("Training vs. Testing:", train_test_overlap)
print("Validation vs. Testing:", val_test_overlap)

print("\nAll leakage checks passed!")
print("All original examples preserved.")
print("Datasets saved successfully.")
