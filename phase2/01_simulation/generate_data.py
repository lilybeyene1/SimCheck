
"""
SIMCHECK Phase 2
Generate synthetic combat training data.
"""

import random
import csv

from pathlib import Path
from combat_env import (
    CombatState,
    step,
    VI_MAX_HP,
    JINX_MAX_HP,
    VI_MAX_MANA,
    JINX_MAX_MANA,
    VI_Q_COOLDOWN,
    JINX_W_COOLDOWN,
    MAX_DISTANCE
)


# 1. Set experiment parameters
# Match Phase 1's dataset size and random seed

NUM_SAMPLES = 10000

rng = random.Random(42)

ACTIONS = ["attack", "ability", "retreat"]
ENEMY_ACTIONS = ["none", "attack", "ability"]


# 2. Define the output location
# Save everything inside the main Phase 2 folder

PHASE_ROOT = Path(__file__).resolve().parent.parent

OUTPUT = PHASE_ROOT / "combat_data.csv"


# 3. Define dataset columns

columns = [
    "player_hp",
    "enemy_hp",
    "distance",
    "player_mana",
    "enemy_mana",
    "player_cooldown",
    "enemy_cooldown",
    "enemy_attack",
    "action",
    "next_player_hp",
    "next_enemy_hp",
    "next_distance",
    "next_player_mana",
    "next_enemy_mana",
    "next_player_cooldown",
    "next_enemy_cooldown"
]


# 4. Generate random combat situations

with open(OUTPUT, "w", newline="") as file:

    writer = csv.writer(file)
    writer.writerow(columns)

    for _ in range(NUM_SAMPLES):

        state = CombatState(
            player_hp=rng.randint(1, VI_MAX_HP),
            enemy_hp=rng.randint(1, JINX_MAX_HP),

            distance=round(
                rng.uniform(0, MAX_DISTANCE), 2
            ),

            player_mana=rng.randint(0, VI_MAX_MANA),
            enemy_mana=rng.randint(0, JINX_MAX_MANA),

            # Abilities are ready in approximately
            # half of the sampled combat states

            player_cooldown=(
                0 if rng.random() < 0.5
                else rng.randint(1, VI_Q_COOLDOWN - 1)
            ),

            enemy_cooldown=(
                0 if rng.random() < 0.5
                else rng.randint(1, JINX_W_COOLDOWN - 1)
            ),

            enemy_attack=rng.choice(ENEMY_ACTIONS)
        )

        action = rng.choice(ACTIONS)

        # Query the ground-truth simulator

        next_state = step(state, action)

        # Save the starting state, selected action
        # and resulting state

        writer.writerow([
            state.player_hp,
            state.enemy_hp,
            state.distance,
            state.player_mana,
            state.enemy_mana,
            state.player_cooldown,
            state.enemy_cooldown,
            state.enemy_attack,
            action,
            next_state.player_hp,
            next_state.enemy_hp,
            next_state.distance,
            next_state.player_mana,
            next_state.enemy_mana,
            next_state.player_cooldown,
            next_state.enemy_cooldown
        ])


# 5. Display the generation summary

print("=" * 55)
print("SIMCHECK PHASE 2 - DATA GENERATION")
print("=" * 55)

print("Generated transitions:", NUM_SAMPLES)
print("Dataset columns:", len(columns))
print("Simulator queries:", NUM_SAMPLES)
print("Saved to:", OUTPUT)

print("\nData generation complete.")
