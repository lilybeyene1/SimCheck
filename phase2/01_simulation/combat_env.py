
"""
SIMCHECK - Phase 2
League-inspired combat simulator: Vi vs Jinx

Champion base statistics: Data Dragon 16.22.1
Ability formulas: League of Legends Wiki

This is a simplified research environment,
not a recreation of the League of Legends engine.
"""

from dataclasses import dataclass
from typing import Literal


# --------------------------------------------------
# 1. Define available actions
# --------------------------------------------------

Action = Literal["attack", "ability", "retreat"]

EnemyAction = Literal["none", "attack", "ability"]


# --------------------------------------------------
# 2. Champion statistics
# --------------------------------------------------

# Vi: level-one base statistics

VI_MAX_HP = 655
VI_MAX_MANA = 295
VI_ATTACK_DAMAGE = 61
VI_ARMOR = 30
VI_MOVEMENT_SPEED = 340
VI_ATTACK_RANGE = 125

# Vi Q: fully charged Vault Breaker

VI_Q_DAMAGE = 100
VI_Q_RANGE = 725
VI_Q_COOLDOWN = 12
VI_Q_COST = 50


# Jinx: level-one base statistics

JINX_MAX_HP = 610
JINX_MAX_MANA = 260
JINX_ATTACK_DAMAGE = 59
JINX_ARMOR = 26
JINX_MOVEMENT_SPEED = 325
JINX_ATTACK_RANGE = 525

# Jinx W: Zap!

JINX_W_DAMAGE = 10 + 1.4 * JINX_ATTACK_DAMAGE
JINX_W_RANGE = 1450
JINX_W_COOLDOWN = 8
JINX_W_COST = 40


# Simplified environment parameters

STEP_DURATION = 1.0
MAX_DISTANCE = 1800.0


# --------------------------------------------------
# 3. Define the combat state
# --------------------------------------------------

@dataclass(frozen=True)
class CombatState:

    player_hp: float
    enemy_hp: float
    distance: float

    player_mana: float
    enemy_mana: float

    player_cooldown: float
    enemy_cooldown: float

    enemy_attack: EnemyAction


# --------------------------------------------------
# 4. Calculate physical damage after armor
# --------------------------------------------------

def physical_damage(raw_damage, armor):

    return raw_damage * 100 / (100 + armor)


# --------------------------------------------------
# 5. Simulate one combat transition
# --------------------------------------------------

def step(state, action: Action):

    if action not in ("attack", "ability", "retreat"):
        raise ValueError("Invalid player action")

    # Copy the starting state

    player_hp = state.player_hp
    enemy_hp = state.enemy_hp

    distance = state.distance

    player_mana = state.player_mana
    enemy_mana = state.enemy_mana

    player_cooldown = state.player_cooldown
    enemy_cooldown = state.enemy_cooldown

    interrupted = False

    # Combat cannot continue if either champion is dead

    if player_hp <= 0 or enemy_hp <= 0:
        return state


    # ----------------------------------------------
    # Vi's action
    # ----------------------------------------------

    if action == "attack":

        # Basic attacks require melee range

        if distance <= VI_ATTACK_RANGE:

            damage = physical_damage(
                VI_ATTACK_DAMAGE,
                JINX_ARMOR
            )

            enemy_hp = max(0, enemy_hp - damage)


    elif action == "ability":

        # Vault Breaker requires mana and readiness

        if (
            player_cooldown == 0
            and player_mana >= VI_Q_COST
        ):

            player_mana -= VI_Q_COST
            player_cooldown = VI_Q_COOLDOWN

            if distance <= VI_Q_RANGE:

                damage = physical_damage(
                    VI_Q_DAMAGE,
                    JINX_ARMOR
                )

                enemy_hp = max(0, enemy_hp - damage)

                # Vi closes the gap when Q connects

                distance = min(distance, VI_ATTACK_RANGE)

                # Approximate Q knockback by interrupting
                # Jinx's pending action for this step

                interrupted = True

            else:

                # Out-of-range Q still moves Vi forward
                # but does not damage Jinx

                distance -= VI_Q_RANGE


    elif action == "retreat":

        # Vi moves away from Jinx

        distance = min(
            MAX_DISTANCE,
            distance + VI_MOVEMENT_SPEED * STEP_DURATION
        )


    # ----------------------------------------------
    # Jinx's response
    # ----------------------------------------------

    # Jinx cannot respond if dead or interrupted

    if enemy_hp > 0 and not interrupted:

        if state.enemy_attack == "attack":

            if distance <= JINX_ATTACK_RANGE:

                damage = physical_damage(
                    JINX_ATTACK_DAMAGE,
                    VI_ARMOR
                )

                player_hp = max(0, player_hp - damage)


        elif state.enemy_attack == "ability":

            if (
                enemy_cooldown == 0
                and enemy_mana >= JINX_W_COST
            ):

                enemy_mana -= JINX_W_COST
                enemy_cooldown = JINX_W_COOLDOWN

                if distance <= JINX_W_RANGE:

                    damage = physical_damage(
                        JINX_W_DAMAGE,
                        VI_ARMOR
                    )

                    player_hp = max(0, player_hp - damage)


    # ----------------------------------------------
    # Update ability cooldowns
    # ----------------------------------------------

    player_cooldown = max(
        0, player_cooldown - STEP_DURATION
    )

    enemy_cooldown = max(
        0, enemy_cooldown - STEP_DURATION
    )


    # Return the resulting combat state

    return CombatState(
        player_hp=player_hp,
        enemy_hp=enemy_hp,
        distance=distance,
        player_mana=player_mana,
        enemy_mana=enemy_mana,
        player_cooldown=player_cooldown,
        enemy_cooldown=enemy_cooldown,
        enemy_attack="none"
    )


# --------------------------------------------------
# 6. Check the environment with an example
# --------------------------------------------------

if __name__ == "__main__":

    initial_state = CombatState(
        player_hp=50,
        enemy_hp=200,
        distance=500,
        player_mana=295,
        enemy_mana=260,
        player_cooldown=0,
        enemy_cooldown=0,
        enemy_attack="ability"
    )

    print("INITIAL STATE")
    print(initial_state)

    for action in ["attack", "ability", "retreat"]:

        print("\nACTION:", action.upper())
        print(step(initial_state, action))
