
"""
SIMCHECK: Decision Regret

Purpose:
Measure how much worse an agent's decision is when it
relies on an imperfect AI world model.

Instead of measuring prediction accuracy alone, we
measure the consequences of trusting those predictions.
"""

from combat_env import CombatState, step
from evaluate_model import predict_outcome, ACTIONS


# Experimental reward weights.
# These are our design choices, not actual League values.

DAMAGE_REWARD = 1
DAMAGE_PENALTY = 1
KILL_REWARD = 100
DEATH_PENALTY = 100

# A regret of at least 20 points will initially be
# considered a decision-critical failure.
CRITICAL_THRESHOLD = 20


def calculate_reward(state, next_player_hp, next_enemy_hp):
    """Assign a numerical score to a combat outcome."""

    # Calculate how much damage each champion received.
    damage_dealt = state.enemy_hp - next_enemy_hp
    damage_taken = state.player_hp - next_player_hp

    # Reward dealing damage and penalize taking damage.
    reward = (
        DAMAGE_REWARD * damage_dealt
        - DAMAGE_PENALTY * damage_taken
    )

    # Give an additional reward for killing the enemy.
    if next_enemy_hp < 0.5:
        reward += KILL_REWARD

    # Give an additional penalty for dying.
    if next_player_hp < 0.5:
        reward -= DEATH_PENALTY

    return reward


def evaluate_decision(state, selected_model="random_forest"):
    """
    Compare the action recommended by our AI
    against the best action in the actual simulator.
    """

    actual_rewards = {}
    predicted_rewards = {}

    # Evaluate every possible action from the same state.
    for action in ACTIONS:

        # Calculate what actually happens.
        actual = step(state, action)

        actual_rewards[action] = calculate_reward(
            state,
            actual.player_hp,
            actual.enemy_hp
        )

        # Ask the AI what it thinks will happen.
        predicted = predict_outcome(
            state, action, selected_model
        )

        predicted_rewards[action] = calculate_reward(
            state,
            predicted[0],
            predicted[1]
        )

    # Find the best action according to the actual simulator.
    best_action = max(actual_rewards, key=actual_rewards.get)

    # Find the best action according to the AI world model.
    model_action = max(
        predicted_rewards,
        key=predicted_rewards.get
    )

    # Measure how much reward is lost by trusting the AI.
    regret = (
        actual_rewards[best_action]
        - actual_rewards[model_action]
    )

    # Check whether the mistake is serious enough to
    # qualify as a decision-critical failure.
    critical_failure = regret >= CRITICAL_THRESHOLD

    return {
        "best_action": best_action,
        "model_action": model_action,
        "actual_rewards": actual_rewards,
        "predicted_rewards": predicted_rewards,
        "regret": regret,
        "critical_failure": critical_failure
    }


# Test our regret calculation on example combat situations.
if __name__ == "__main__":

    scenarios = [
        CombatState(20, 40, 1, "skillshot", 0),
        CombatState(20, 40, 1, "skillshot", 2),
        CombatState(20, 10, 1, "basic", 0),
        CombatState(100, 100, 5, "none", 0)
    ]

    for i, state in enumerate(scenarios, start=1):

        result = evaluate_decision(state)

        print(f"\n{'=' * 50}")
        print(f"SCENARIO {i}")
        print(state)

        print("\nActual vs. Predicted Rewards:")

        for action in ACTIONS:
            actual = result["actual_rewards"][action]
            predicted = result["predicted_rewards"][action]

            print(
                f"{action:8s}: "
                f"actual={actual:.2f}, "
                f"predicted={predicted:.2f}"
            )

        print("\nActual best action:", result["best_action"])
        print("AI recommended action:", result["model_action"])
        print("Decision regret:", result["regret"])
        print("Critical failure:", result["critical_failure"])
