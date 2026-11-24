
"""
SIMCHECK: Budget-Constrained Testing Experiments

Compare all five testing strategies using frozen
rankings and the ground-truth reference dataset.
"""

import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Find the main Phase 1 folder
PHASE_ROOT = Path(__file__).resolve().parent.parent

# Point to our existing experiment and results folders
EXPERIMENTS = PHASE_ROOT / "experiments"
RESULTS = PHASE_ROOT / "results"

# STEP 1: Verify that the rankings have not changed

path = EXPERIMENTS/ "frozen_rankings.csv"

with open(path, "rb") as file:
    fingerprint = hashlib.sha256(
        file.read()
    ).hexdigest()

with open(EXPERIMENTS / "rankings_sha256.txt") as file:
    expected = file.read().strip()

assert fingerprint == expected

rankings = pd.read_csv(path)


# STEP 2: Define our experimental settings

MODELS = ["random_forest", "neural_network"]

METHODS = [
    "random",
    "uncertainty",
    "estimated_error",
    "action_sensitivity",
    "raas"
]

BUDGETS = [30, 75, 150, 300]

all_results = []


# STEP 3: Evaluate the testing strategies

for model in MODELS:

    truth = pd.read_csv(
        EXPERIMENTS / f"{model}_oracle_decisions.csv" #This loads the existing ground truth file for whichever model is being evaluated
    )

    total_failures = truth["critical_failure"].sum() #total number of failures available to discover
 # filters frozen rankings file which contains both world models to the current modelneing tested
    model_rankings = rankings[
        rankings["model"] == model
    ]

    for method in METHODS:
       # model identifies which state it considers most important to test
        selected = model_rankings[
            model_rankings["method"] == method
        ]
        # groups the rankings by their repetition number since repeated trials lets us measure how much each result varies
        for repetition, group in selected.groupby(
            "repetition"
        ):

            group = group.sort_values("rank") # rank the states in order of importance for testing

            for budget in BUDGETS: #How many critical failures can each strategy discover when allowed only "budget" simulator queries?

                # Each verified state costs three queries

                states_allowed = budget // 3 #note: Each combat state requires three simulator queries

                tested = group.head(states_allowed) #selects the highest-ranked states that fit within the budget

               # Check that the selection is valid
                assert len(tested) == states_allowed #ensures I selected exactly the number of states allowed
                assert tested["state_id"].is_unique #ensures I didn't accidentally select the same state twice

                # Reveal only the selected states'
                # ground-truth outcomes

                outcomes = tested.merge(
                    truth,
                    on="state_id",
                    validate="one_to_one"
                )

                assert len(outcomes) == states_allowed

                failures = outcomes[
                    "critical_failure"
                ].sum()

                all_results.append({
                    "model": model,
                    "method": method,
                    "repetition": repetition,
                    "query_budget": budget,
                    "states_tested": states_allowed,
                    "critical_failures": failures,
                    "failure_yield": failures / states_allowed,
                    "failure_coverage": failures / total_failures
                    if total_failures > 0 else np.nan,
                    "cumulative_regret": outcomes["regret"].sum(),
                    "mean_regret": outcomes["regret"].mean(),
                    "avoidable_deaths": outcomes[
                        "avoidable_death"
                    ].sum()
                })


# STEP 4: Save individual experimental results

results = pd.DataFrame(all_results)

results.to_csv(
    RESULTS / "budget_experiment_trials.csv",
    index=False
)


# STEP 5: Summarize performance

summary = results.groupby([
    "model",
    "method",
    "query_budget"
]).agg(
    mean_failures=("critical_failures", "mean"),
    mean_yield=("failure_yield", "mean"),
    mean_coverage=("failure_coverage", "mean"),
    cumulative_regret=("cumulative_regret", "mean"),
    mean_regret=("mean_regret", "mean"),
    avoidable_deaths=("avoidable_deaths", "mean")
).reset_index()

summary.to_csv(
    RESULTS / "budget_experiment_summary.csv",
    index=False
)


# STEP 6: Calculate the random baseline's
# empirical 95% outcome interval

random_results = results[
    results["method"] == "random"
]

random_intervals = random_results.groupby([
    "model",
    "query_budget"
])["critical_failures"].agg(
    lower=lambda x: x.quantile(0.025),
    upper=lambda x: x.quantile(0.975)
).reset_index()

random_intervals.to_csv(
    RESULTS / "random_baseline_intervals.csv",
    index=False
)


# STEP 7: Print our primary research results

for model in MODELS:

    print("\n" + "=" * 60)
    print(model.upper())
    print("=" * 60)

    subset = summary[summary["model"] == model]

    table = subset.pivot(
        index="method",
        columns="query_budget",
        values="mean_failures"
    )

    print("\nCritical failures discovered:")
    print(table.round(2).to_string())

    print("\nRandom baseline outcome intervals:")

    print(
        random_intervals[
            random_intervals["model"] == model
        ].to_string(index=False)
    )

    print("\nResults at maximum query budget:")

    print(
        subset[subset["query_budget"] == 300][[
            "method",
            "mean_failures",
            "mean_yield",
            "mean_coverage",
            "cumulative_regret",
            "avoidable_deaths"
        ]].round(4).to_string(index=False)
    )


    # STEP 8: Plot failures discovered
    # as the query budget increases

    plt.figure(figsize=(9, 6))

    for method in METHODS:

        curve = subset[
            subset["method"] == method
        ].sort_values("query_budget")

        plt.plot(
            [0] + curve["query_budget"].tolist(),
            [0] + curve["mean_failures"].tolist(),
            marker="o",
            label=method
        )

    plt.xlabel("Simulator Query Budget")
    plt.ylabel("Critical Failures Discovered")
    plt.title(f"SIMCHECK: {model.replace('_', ' ').title()}")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()

    plt.savefig(
        RESULTS/ f"{model}_failure_discovery.png",
        dpi=300
    )

    plt.close()


print("\nTotal experimental trials:", len(results))
print("Additional simulator calls: 0")
print("Budget-constrained experiment complete.")
