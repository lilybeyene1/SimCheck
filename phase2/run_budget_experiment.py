
"""
SIMCHECK Phase 2
Compare five testing strategies under equal
ground-truth simulator-query budgets.
"""

import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


# 1. Define experimental parameters.

ROOT = Path(__file__).resolve().parent

EXPERIMENTS = ROOT / "experiments"
RESULTS = ROOT / "results"

MODELS = ["random_forest", "neural_network"]

QUERY_BUDGETS = [30, 75, 150, 300]

QUERIES_PER_STATE = 3

METHODS = [
    "random",
    "uncertainty",
    "estimated_error",
    "action_sensitivity",
    "raas"
]

LABELS = {
    "random": "Random",
    "uncertainty": "Uncertainty",
    "estimated_error": "Estimated Error",
    "action_sensitivity": "Action Sensitivity",
    "raas": "RAAS"
}


# 2. Verify the frozen rankings.

rankings_path = EXPERIMENTS / "frozen_rankings.csv"

expected_hash = (
    EXPERIMENTS / "rankings_sha256.txt"
).read_text().strip()

actual_hash = hashlib.sha256(
    rankings_path.read_bytes()
).hexdigest()

assert actual_hash == expected_hash

rankings = pd.read_csv(rankings_path)

print("Frozen rankings verified.")

records = []


# 3. Evaluate each world model.

for model_name in MODELS:

    oracle = pd.read_csv(
        EXPERIMENTS
        / f"{model_name}_oracle_decisions.csv"
    )

    total_failures = oracle["critical_failure"].sum()

    model_rankings = rankings[
        rankings["model"] == model_name
    ]

    print("\nMODEL:", model_name)
    print("Total critical failures:", total_failures)


    # 4. Evaluate every frozen ranking.

    for (method, repetition), group in (
        model_rankings.groupby(["method", "repetition"])
    ):

        group = group.sort_values("rank")


        # 5. Apply each simulator-query budget.

        for budget in QUERY_BUDGETS:

            states_tested = budget // QUERIES_PER_STATE

            selected = group.head(states_tested)

            evaluated = selected.merge(
                oracle,
                on="state_id",
                validate="one_to_one"
            )

            assert len(evaluated) == states_tested

            failures = int(
                evaluated["critical_failure"].sum()
            )

            # Failure yield:
            # Proportion of tested states that fail.

            failure_yield = failures / states_tested

            # Failure coverage:
            # Proportion of all existing failures found.

            failure_coverage = (
                failures / total_failures
                if total_failures > 0
                else np.nan
            )

            records.append({
                "model": model_name,
                "method": method,
                "repetition": repetition,
                "query_budget": budget,
                "states_tested": states_tested,
                "critical_failures": failures,
                "failure_yield": failure_yield,
                "failure_coverage": failure_coverage,
                "cumulative_regret":
                    evaluated["regret"].sum(),
                "mean_regret":
                    evaluated["regret"].mean(),
                "avoidable_deaths":
                    evaluated["avoidable_death"].sum()
            })


# 6. Save individual experimental trials.

trials = pd.DataFrame(records)

assert len(trials) == 832

trials.to_csv(
    RESULTS / "budget_experiment_trials.csv",
    index=False
)


# 7. Summarize the experimental results.
# Average across the 100 random repetitions.

summary = trials.groupby(
    ["model", "method", "query_budget"],
    as_index=False
).agg(
    mean_failures=("critical_failures", "mean"),
    mean_yield=("failure_yield", "mean"),
    mean_coverage=("failure_coverage", "mean"),
    cumulative_regret=("cumulative_regret", "mean"),
    mean_regret=("mean_regret", "mean"),
    avoidable_deaths=("avoidable_deaths", "mean")
)

summary.to_csv(
    RESULTS / "budget_experiment_summary.csv",
    index=False
)


# 8. Calculate empirical random-baseline intervals.
# These describe variation across random selections
# within the fixed candidate pool.

random_trials = trials[
    trials["method"] == "random"
]

intervals = random_trials.groupby(
    ["model", "query_budget"]
)["critical_failures"].agg(
    lower=lambda x: x.quantile(0.025),
    upper=lambda x: x.quantile(0.975)
).reset_index()

intervals.to_csv(
    RESULTS / "random_baseline_intervals.csv",
    index=False
)


# 9. Plot critical failure discovery.

for model_name in MODELS:

    model_summary = summary[
        summary["model"] == model_name
    ]

    plt.figure(figsize=(9, 6))

    for method in METHODS:

        data = model_summary[
            model_summary["method"] == method
        ].sort_values("query_budget")

        plt.plot(
            data["query_budget"],
            data["mean_failures"],
            marker="o",
            label=LABELS[method]
        )

    band = intervals[
        intervals["model"] == model_name
    ].sort_values("query_budget")

    plt.fill_between(
        band["query_budget"].to_numpy(),
        band["lower"].to_numpy(),
        band["upper"].to_numpy(),
        alpha=0.12,
        color="gray",
        label="Random 95% empirical interval"
    )

    plt.xlabel("Simulator Query Budget")
    plt.ylabel("Critical Failures Discovered")

    plt.title(
        f"Phase 2 Failure Discovery: "
        f"{model_name.replace('_', ' ').title()}"
    )

    plt.xticks(QUERY_BUDGETS)
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()

    plt.savefig(
        RESULTS
        / f"{model_name}_failure_discovery.png",
        dpi=300
    )

    plt.close()


# 10. Display the final results.

for model_name in MODELS:

    data = summary[
        summary["model"] == model_name
    ]

    comparison = data.pivot(
        index="method",
        columns="query_budget",
        values="mean_failures"
    )

    print("\n" + "=" * 65)
    print(model_name.upper())
    print("CRITICAL FAILURES DISCOVERED")
    print("=" * 65)

    print(comparison.round(2))

    print("\nRESULTS AT MAXIMUM BUDGET")

    print(
        data[data["query_budget"] == 300]
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )


print("\n" + "=" * 65)
print("Experimental trials:", len(trials))
print("Additional simulator queries: 0")
print("Phase 2 budget experiment complete.")
