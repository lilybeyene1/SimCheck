
"""
SIMCHECK: Combat Training Data Generator

Purpose:
Generate examples of combat situations and their outcomes.

Each row contains:
1. The starting combat state
2. The action the player chose
3. The actual outcome from our combat simulator

These examples will be used to train an AI world model.

Important:
The AI will learn from this dataset, not from directly reading
the combat rules in combat_env.py.
"""

import csv
import random

# Import our combat environment.
# This lets us use the rules we already created.
from combat_env import CombatState, step

# Possible player actions and enemy attack types.
ACTIONS = ["attack", "dodge", "retreat"]
ENEMY_ATTACKS = ["none", "basic", "skillshot"]


def generate_dataset(n=10000, seed=42):
    """
    Generate n combat transitions.

    n = number of examples to create
    seed = makes the random data reproducible

    Using the same seed produces the same dataset,
    which is important for repeatable research experiments.
    """

    # Create a random number generator with a fixed seed.
    rng = random.Random(seed)

    # Store each generated example here.
    rows = []

    for _ in range(n):

        # STEP 1: Create a random starting combat situation.
        state = CombatState(
            player_hp=rng.randint(5, 100),
            enemy_hp=rng.randint(5, 100),
            distance=rng.randint(1, 5),
            enemy_attack=rng.choice(ENEMY_ATTACKS),
            dodge_cooldown=rng.randint(0, 2)
        )

        # STEP 2: Randomly choose one player action.
        # Right now, all three actions have equal probability.
        action = rng.choice(ACTIONS)

        # STEP 3: Use our simulator to calculate the real outcome.
        # This gives us the correct answer for training.
        next_state = step(state, action)

        # STEP 4: Save the starting state, action, and outcome.
        rows.append({
            "player_hp": state.player_hp,
            "enemy_hp": state.enemy_hp,
            "distance": state.distance,
            "enemy_attack": state.enemy_attack,
            "dodge_cooldown": state.dodge_cooldown,
            "action": action,
            "next_player_hp": next_state.player_hp,
            "next_enemy_hp": next_state.enemy_hp,
            "next_distance": next_state.distance,
            "next_dodge_cooldown": next_state.dodge_cooldown
        })

    return rows


# Run this section only when generate_data.py is executed directly.
if __name__ == "__main__":

    # Generate 10,000 training examples.
    data = generate_dataset()

    # Save the generated examples as a CSV file.
    # The first row contains the column names.
    with open("combat_data.csv", "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=data[0].keys()
        )

        writer.writeheader()
        writer.writerows(data)

    # Confirm the file was created successfully.
    print(f"Generated {len(data)} combat transitions.")
    print("Saved to combat_data.csv")

    # Display one example to inspect the data format.
    print("\nFirst example:")
    print(data[0])
