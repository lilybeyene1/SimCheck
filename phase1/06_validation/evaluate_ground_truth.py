
"""
SIMCHECK: Ground-Truth Evaluation

Evaluate every candidate state using the actual
combat simulator and calculate decision regret.

Basically for each situation, we'll determine:
- Which action the AI recommends.
- Which action is actually optimal.
- How much reward is lost by following the AI's recommendation (decision regret).
- Whether that mistake qualifies as a decision-critical failure (regret ≥ 20).

This reference dataset is for evaluation only.
Testing strategies must not access it.
"""

import sys
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


# Find the main Phase 1 folder
PHASE_ROOT = Path(__file__).resolve().parent.parent

# Allow imports directly from the simulation folder
sys.path.insert(0, str(PHASE_ROOT / "01_simulation"))

from combat_env import CombatState, step


# Point to our existing folders
EXPERIMENTS = PHASE_ROOT / "experiments"
DATASETS = PHASE_ROOT / "datasets"
RESULTS = PHASE_ROOT / "results"


# STEP 1: Verify our frozen experimental rankings

with open(EXPERIMENTS / "frozen_rankings.csv", "rb") as file:

    fingerprint = hashlib.sha256(
        file.read()
    ).hexdigest()

with open(EXPERIMENTS / "rankings_sha256.txt") as file:

    expected = file.read().strip()

assert fingerprint == expected

print("Frozen rankings verified successfully.")


# STEP 2: Configure the experiment

ACTIONS = ["attack", "dodge", "retreat"]

CRITICAL_THRESHOLD = 20

pool = pd.read_csv(DATASETS / "candidate_pool.csv")

simulator_queries = 0

actual_outcomes = []


# STEP 3: Define our original reward function

def calculate_reward(state, next_state):

    damage_dealt = state.enemy_hp - next_state.enemy_hp

    damage_taken = state.player_hp - next_state.player_hp

    reward = damage_dealt - damage_taken

    if next_state.enemy_hp < 0.5:
        reward += 100

    if next_state.player_hp < 0.5:
        reward -= 100

    return reward


# STEP 4: Reveal the actual outcomes
#
# Each starting state requires three simulator calls

for row in pool.itertuples(index=False):

    state = CombatState(
        player_hp=row.player_hp,
        enemy_hp=row.enemy_hp,
        distance=row.distance,
        enemy_attack=row.enemy_attack,
        dodge_cooldown=row.dodge_cooldown
    )

    for action in ACTIONS:

        outcome = step(state, action)

        simulator_queries += 1

        actual_outcomes.append({
            "state_id": row.state_id,
            "action": action,
            "actual_reward": calculate_reward(state, outcome),
            "actual_next_player_hp": outcome.player_hp,
            "actual_next_enemy_hp": outcome.enemy_hp,
            "actual_next_distance": outcome.distance,
            "actual_next_dodge_cooldown": outcome.dodge_cooldown
        })


truth = pd.DataFrame(actual_outcomes)

assert simulator_queries == len(pool) * len(ACTIONS)

truth.to_csv(
    EXPERIMENTS / "oracle_outcomes.csv",
    index=False
)


# STEP 5: Evaluate each world model

for name in ["random_forest", "neural_network"]:

    predictions = pd.read_csv(
        RESULTS / f"{name}_candidate_predictions.csv"
    )

    combined = predictions.merge(
        truth,
        on=["state_id", "action"],
        validate="one_to_one"
    )

    assert len(combined) == len(truth)

    decisions = []


    # STEP 6: Identify the recommended action
    # and compare it with the optimal action

    for state_id, group in combined.groupby("state_id"):

        group = group.set_index("action").loc[ACTIONS]

        best_action = group["actual_reward"].idxmax()

        model_action = group["predicted_reward"].idxmax()

        optimal_reward = group.loc[
            best_action, "actual_reward"
        ]

        achieved_reward = group.loc[
            model_action, "actual_reward"
        ]

        regret = optimal_reward - achieved_reward

        model_died = (
            group.loc[
                model_action, "actual_next_player_hp"
            ] < 0.5
        )

        survival_possible = (
            group["actual_next_player_hp"] >= 0.5
        ).any()

        decisions.append({
            "state_id": state_id,
            "optimal_action": best_action,
            "model_action": model_action,
            "optimal_reward": optimal_reward,
            "achieved_reward": achieved_reward,
            "regret": regret,
            "critical_failure": regret >= CRITICAL_THRESHOLD,
            "avoidable_death": bool(
                model_died and survival_possible
            )
        })


    # STEP 7: Save the decision-level results

    results = pd.DataFrame(decisions)

    assert (results["regret"] >= 0).all()

    results.to_csv(
        EXPERIMENTS / f"{name}_oracle_decisions.csv",
        index=False
    )


    # STEP 8: Report overall decision reliability

    print("\n" + "=" * 55)
    print(name.upper())
    print("=" * 55)

    print("States evaluated:", len(results))

    print(
        "Optimal decisions:",
        (results["regret"] == 0).sum()
    )

    print(
        "Optimal decision rate:",
        round((results["regret"] == 0).mean() * 100, 2),
        "%"
    )

    print(
        "Critical failures:",
        results["critical_failure"].sum()
    )

    print(
        "Critical failure rate:",
        round(results["critical_failure"].mean() * 100, 2),
        "%"
    )

    print("Avoidable deaths:", results["avoidable_death"].sum())

    print("Mean regret:", round(results["regret"].mean(), 4))

    print("Maximum regret:", results["regret"].max())

    print("\nModel action distribution:")
    print(results["model_action"].value_counts())


print("\nTotal oracle simulator queries:", simulator_queries)
print("Ground-truth evaluation complete.")
