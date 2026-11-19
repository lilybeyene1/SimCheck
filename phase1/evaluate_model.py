
"""
SIMCHECK: Initial World Model Evaluation

Purpose:
Compare predicted combat outcomes with actual outcomes.

We test attack, dodge, and retreat from the SAME starting state.

This helps us identify situations where the model
misunderstands how different actions affect combat.

This is an initial diagnostic, not yet our formal
budget-constrained testing experiment.
"""

import joblib
import pandas as pd

from combat_env import CombatState, step


# STEP 1: Load our trained Random Forest world model.
# STEP 1: Load both trained world models.
saved = joblib.load("models/random_forest.joblib")

neural_network = joblib.load(
    "models/neural_network.joblib"
)

model = saved["model"]
features = saved["features"]
targets = saved["targets"]

ACTIONS = ["attack", "dodge", "retreat"]


def predict_outcome(state, action, selected_model="random_forest"):
    """
    Predict the next combat state using our trained model.

    The model does not directly access the simulator rules.
    """

    # Describe the starting situation and chosen action.
    row = {
        "player_hp": state.player_hp,
        "enemy_hp": state.enemy_hp,
        "distance": state.distance,
        "dodge_cooldown": state.dodge_cooldown,
        "enemy_attack": state.enemy_attack,
        "action": action
    }

    # Convert text categories into numeric features.
    encoded = pd.get_dummies(
        pd.DataFrame([row]),
        columns=["enemy_attack", "action"],
        dtype=int
    )

    # Select which trained world model to use.
    if selected_model == "random_forest":
        active_model = saved

    elif selected_model == "neural_network":
        active_model = neural_network

    else:
        raise ValueError("Unknown world model")

    # Match the feature columns used to train that model.
    encoded = encoded.reindex(
        columns=active_model["features"],
        fill_value=0
    )

    # Ask the selected model to predict the outcome.
    return active_model["model"].predict(encoded)[0]


def evaluate_scenario(state):
    """Compare actual and predicted results for each action."""

    print(f"\nStarting state: {state}")

    for action in ACTIONS:

        # Ground truth: what actually happens in the simulator.
        actual = step(state, action)

        # AI prediction: what the learned model expects.
        predicted = predict_outcome(state, action)

        actual_values = [
            actual.player_hp,
            actual.enemy_hp,
            actual.distance,
            actual.dodge_cooldown
        ]

        print(f"\nAction: {action.upper()}")

        for name, true_value, prediction in zip(
            targets, actual_values, predicted
        ):
            error = abs(true_value - prediction)

            print(
                f"  {name}: "
                f"actual={true_value}, "
                f"predicted={prediction:.2f}, "
                f"error={error:.2f}"
            )


# STEP 2: Define some important combat situations.
scenarios = [
    # Low health, incoming skillshot, dodge available.
    CombatState(20, 40, 1, "skillshot", 0),

    # Low health, incoming skillshot, dodge unavailable.
    CombatState(20, 40, 1, "skillshot", 2),

    # Enemy nearly dead; attacking could finish them.
    CombatState(20, 10, 1, "basic", 0),

    # High health, no immediate enemy attack.
    CombatState(100, 100, 5, "none", 0)
]


# STEP 3: Evaluate all actions in each situation.
if __name__ == "__main__":
    for i, state in enumerate(scenarios, start=1):
        print(f"\n{'=' * 55}")
        print(f"SCENARIO {i}")
        print(f"{'=' * 55}")

        evaluate_scenario(state)