
"""
SIMCHECK: MOBA Combat Environment

Purpose:
Create a simplified League of Legends-inspired combat simulator.

This simulator represents the "ground truth" of our experiment.
It tells us what ACTUALLY happens when a player takes an action.

Later, we'll train an AI world model to predict these outcomes
and compare its predictions against the real simulator.

Note: These are simplified combat rules, not actual LoL mechanics.
"""

from dataclasses import dataclass
from typing import Literal

# Define the actions our player can choose
# Literal restricts the expected action values to these three strings
Action = Literal["attack", "dodge", "retreat"]

# Define the possible types of incoming enemy attacks
EnemyAttack = Literal["none", "basic", "skillshot"]


@dataclass(frozen=True)
class CombatState:
    # A CombatState stores everything needed to describe
    # the current combat situation before an action is taken

    player_hp: int          # Player's remaining health
    enemy_hp: int           # Enemy's remaining health
    distance: int           # Distance between player and enemy (1 to 5)
    enemy_attack: EnemyAttack  # Type of incoming enemy attack
    dodge_cooldown: int     # Turns until dodge is available again


def step(state: CombatState, action: Action) -> CombatState:
    """
    Simulate one turn of combat.

    Input:
        state  = current combat situation
        action = player's chosen action

    Output:
        A new CombatState showing what happened after the action.

    This function is our ground-truth simulator.
    """

    # Copy the current values so we can update them
    player_hp = state.player_hp
    enemy_hp = state.enemy_hp
    distance = state.distance
    cooldown = state.dodge_cooldown

    # STEP 1: The player acts first

    if action == "attack":
        # Basic attacks only reach enemies within distance 2
        # A successful attack deals 15 damage
        if distance <= 2:
            enemy_hp = max(0, enemy_hp - 15)

    elif action == "retreat":
        # Retreat moves the player one unit farther away
        # Distance cannot exceed 5
        distance = min(5, distance + 1)

    elif action == "dodge" and cooldown == 0:
        # A ready dodge activates and starts a cooldown
        # Whether it prevents damage depends on the enemy attack
        cooldown = 2

    # STEP 2: The enemy responds, but only if still alive

    if enemy_hp > 0:

        if state.enemy_attack == "basic":
            # Basic attacks deal 10 damage within distance 1
            # Dodging does not prevent basic attack damage
            if distance <= 1:
                player_hp = max(0, player_hp - 10)

        elif state.enemy_attack == "skillshot":
            # Skillshots deal 25 damage within distance 3
            if distance <= 3:

                # A skillshot is avoided only if the player
                # chose dodge AND dodge was ready at the start
                successful_dodge = (
                    action == "dodge"
                    and state.dodge_cooldown == 0
                )

                if not successful_dodge:
                    player_hp = max(0, player_hp - 25)

    # STEP 3: Update the cooldown after the turn
    # A dodge activated this turn will end with cooldown 1
    cooldown = max(0, cooldown - 1)

    # Return the updated state
    # Enemy attack resets because this is a one-turn simulator
    return CombatState(
        player_hp=player_hp,
        enemy_hp=enemy_hp,
        distance=distance,
        enemy_attack="none",
        dodge_cooldown=cooldown
    )


# This section runs only when we execute combat_env.py directly
# It does not run when another file imports the step() function
if __name__ == "__main__":

    # Four example situations to test our combat rules
    scenarios = [
        CombatState(20, 40, 1, "skillshot", 0),  # Dodge available
        CombatState(20, 40, 1, "skillshot", 2),  # Dodge unavailable
        CombatState(20, 40, 3, "skillshot", 0),  # Farther away
        CombatState(20, 10, 1, "basic", 0)       # Enemy nearly dead
    ]

    # Try every available action in each scenario
    # This helps us see when different choices change the outcome
    for i, state in enumerate(scenarios, start=1):
        print(f"\nScenario {i}: {state}")

        for action in ["attack", "dodge", "retreat"]:
            result = step(state, action)
            print(f"  {action:8s} -> {result}")
