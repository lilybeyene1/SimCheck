
"""
SIMCHECK: World Model Decision Comparison

Purpose:
Investigate whether improvements in prediction accuracy
also translate into improvements in combat decisions.

Both models are evaluated using identical combat scenarios,
reward functions, and critical failure thresholds.
"""

import sys
from pathlib import Path


# Find the main Phase 1 folder
PHASE_ROOT = Path(__file__).resolve().parent.parent

# Allow imports from the simulation and failure detection folders
sys.path.insert(0, str(PHASE_ROOT / "01_simulation"))
sys.path.insert(0, str(PHASE_ROOT / "04_failure_detection"))

from combat_env import CombatState
from decision_regret import evaluate_decision


# STEP 1: Identify the two world models being compared

models = ["random_forest", "neural_network"]


# STEP 2: Reuse our original four combat scenarios

scenarios = [
    CombatState(20, 40, 1, "skillshot", 0),
    CombatState(20, 40, 1, "skillshot", 2),
    CombatState(20, 10, 1, "basic", 0),
    CombatState(100, 100, 5, "none", 0)
]


# STEP 3: Create a place to store each model's results

summary = {
    name: {
        "total_regret": 0,
        "critical_failures": 0,
        "optimal_decisions": 0
    }
    for name in models
}


# STEP 4: Evaluate both models on every combat scenario

for i, state in enumerate(scenarios, start=1):

    print("\n" + "=" * 55)
    print(f"SCENARIO {i}")
    print(state)

    for name in models:

        # Evaluate the selected model's decision
        result = evaluate_decision(state, name)

        print(f"\nModel: {name}")

        print("Actual best action:", result["best_action"])

        print("Recommended action:", result["model_action"])

        print("Decision regret:", result["regret"])

        print("Critical failure:", result["critical_failure"])

        # Display predicted rewards for all possible actions
        print("Predicted rewards:")

        for action, reward in result["predicted_rewards"].items():
            print(f"  {action}: {reward:.2f}")

        # Record the total regret accumulated by this model
        summary[name]["total_regret"] += result["regret"]

        # Count decision-critical failures
        if result["critical_failure"]:
            summary[name]["critical_failures"] += 1

        # Zero regret means the chosen action was optimal,
        # even when multiple actions share the highest reward
        if result["regret"] == 0:
            summary[name]["optimal_decisions"] += 1


# STEP 5: Print the final comparison

print("\n" + "=" * 55)
print("FINAL MODEL COMPARISON")
print("=" * 55)

for name in models:

    results = summary[name]

    print(f"\nModel: {name}")

    print("Total regret:", results["total_regret"])

    print(
        "Average regret:",
        results["total_regret"] / len(scenarios)
    )

    print(
        "Critical failures:",
        results["critical_failures"]
    )

    print(
        "Optimal decisions:",
        f'{results["optimal_decisions"]}/{len(scenarios)}'
    )
