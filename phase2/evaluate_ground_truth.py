
"""
SIMCHECK Phase 2
Evaluate ground-truth combat outcomes and identify
decision-critical world-model failures.
"""

import hashlib
import numpy as np
import pandas as pd

from pathlib import Path

from combat_env import (
    CombatState,
    step,
    VI_MAX_HP,
    JINX_MAX_HP
)


# 1. Define experimental parameters.

ROOT = Path(__file__).resolve().parent

EXPERIMENTS = ROOT / "experiments"

MODELS = ["random_forest", "neural_network"]

ACTIONS = ["attack", "ability", "retreat"]

CRITICAL_THRESHOLD = 20

KILL_REWARD = 100
DEATH_PENALTY = 100


# 2. Verify that the experimental rankings
# have not changed since they were frozen.

rankings_path = EXPERIMENTS / "frozen_rankings.csv"

expected_hash = (
    EXPERIMENTS / "rankings_sha256.txt"
).read_text().strip()

actual_hash = hashlib.sha256(
    rankings_path.read_bytes()
).hexdigest()

assert actual_hash == expected_hash, (
    "Frozen rankings have changed!"
)

print("Frozen rankings verified.")


# 3. Calculate actual combat utility.
#
# Health changes are normalized to percentage
# points of each champion's maximum health.
#
# Actual deaths require zero remaining HP.

def calculate_actual_reward(state, next_state):

    damage_dealt = (
        (state.enemy_hp - next_state.enemy_hp)
        / JINX_MAX_HP * 100
    )

    damage_received = (
        (state.player_hp - next_state.player_hp)
        / VI_MAX_HP * 100
    )

    reward = damage_dealt - damage_received

    if next_state.enemy_hp <= 0:
        reward += KILL_REWARD

    if next_state.player_hp <= 0:
        reward -= DEATH_PENALTY

    return reward


# 4. Load the unseen candidate states.

pool = pd.read_csv(
    ROOT / "datasets/candidate_pool.csv"
)

records = []

query_count = 0

print("\nGENERATING GROUND-TRUTH OUTCOMES")


# 5. Query the simulator for every possible action.

for row in pool.itertuples(index=False):

    state = CombatState(
        player_hp=row.player_hp,
        enemy_hp=row.enemy_hp,
        distance=row.distance,
        player_mana=row.player_mana,
        enemy_mana=row.enemy_mana,
        player_cooldown=row.player_cooldown,
        enemy_cooldown=row.enemy_cooldown,
        enemy_attack=row.enemy_attack
    )

    for action in ACTIONS:

        next_state = step(state, action)

        query_count += 1

        reward = calculate_actual_reward(
            state,
            next_state
        )

        records.append({
            "state_id": row.state_id,
            "action": action,
            "actual_reward": reward,
            "actual_next_player_hp": next_state.player_hp,
            "actual_next_enemy_hp": next_state.enemy_hp,
            "actual_next_distance": next_state.distance,
            "actual_next_player_mana": next_state.player_mana,
            "actual_next_enemy_mana": next_state.enemy_mana,
            "actual_next_player_cooldown":
                next_state.player_cooldown,
            "actual_next_enemy_cooldown":
                next_state.enemy_cooldown
        })


# 6. Save the exhaustive ground-truth reference.

oracle = pd.DataFrame(records)

assert query_count == 15000

assert len(oracle) == 15000

assert (
    oracle.groupby("state_id").size() == 3
).all()

oracle.to_csv(
    EXPERIMENTS / "oracle_outcomes.csv",
    index=False
)

print("Candidate states:", len(pool))
print("Ground-truth outcomes:", len(oracle))
print("Simulator queries:", query_count)


# 7. Evaluate each learned world model.

for model_name in MODELS:

    predictions = pd.read_csv(
        ROOT / "results"
        / f"{model_name}_candidate_predictions.csv"
    )

    combined = predictions.merge(
        oracle,
        on=["state_id", "action"],
        validate="one_to_one"
    )

    assert len(combined) == 15000

    decisions = []

    print("\nMODEL:", model_name)


    # 8. Compare predicted and optimal actions.

    for state_id, group in combined.groupby("state_id"):

        # Preserve the same deterministic tie-breaking
        # order for predicted and actual actions.

        group = group.set_index("action").loc[ACTIONS]

        optimal_action = group["actual_reward"].idxmax()

        model_action = group["predicted_reward"].idxmax()

        optimal_reward = group.loc[
            optimal_action, "actual_reward"
        ]

        achieved_reward = group.loc[
            model_action, "actual_reward"
        ]


        # 9. Calculate decision regret.
        # Regret measures the actual utility lost
        # by following the learned model's decision.

        regret = float(
            optimal_reward - achieved_reward
        )

        assert regret >= -1e-9

        regret = max(0.0, regret)

        critical_failure = (
            regret >= CRITICAL_THRESHOLD
        )


        # 10. Identify avoidable deaths.
        # The selected action kills Vi, but at least
        # one alternative action allows her to survive.

        chosen_hp = group.loc[
            model_action, "actual_next_player_hp"
        ]

        alternative_survival = (
            group["actual_next_player_hp"] > 0
        ).any()

        avoidable_death = (
            chosen_hp <= 0
            and alternative_survival
        )

        decisions.append({
            "state_id": state_id,
            "optimal_action": optimal_action,
            "model_action": model_action,
            "optimal_reward": optimal_reward,
            "achieved_reward": achieved_reward,
            "regret": regret,
            "critical_failure": critical_failure,
            "avoidable_death": avoidable_death
        })


    # 11. Save state-level decision outcomes.

    results = pd.DataFrame(decisions)

    assert len(results) == 5000

    results.to_csv(
        EXPERIMENTS
        / f"{model_name}_oracle_decisions.csv",
        index=False
    )


    # 12. Summarize decision quality.
    # Zero-regret decisions count as optimal,
    # even when multiple actions are equally good.

    optimal_count = np.isclose(
        results["regret"],
        0,
        atol=1e-9
    ).sum()

    critical_count = results["critical_failure"].sum()

    death_count = results["avoidable_death"].sum()

    print("Optimal decisions:", optimal_count)

    print(
        "Optimal decision rate:",
        f"{optimal_count / len(results) * 100:.2f}%"
    )

    print(
        "Suboptimal decisions:",
        len(results) - optimal_count
    )

    print("Critical failures:", critical_count)

    print(
        "Critical failure rate:",
        f"{critical_count / len(results) * 100:.2f}%"
    )

    print("Avoidable deaths:", death_count)

    print(
        "Mean decision regret:",
        round(results["regret"].mean(), 6)
    )

    print(
        "Maximum decision regret:",
        round(results["regret"].max(), 6)
    )

    print("\nModel action distribution:")

    print(results["model_action"].value_counts())


print("\n" + "=" * 55)
print("TOTAL GROUND TRUTH QUERIES:", query_count)
print("Ground-truth evaluation complete.")
