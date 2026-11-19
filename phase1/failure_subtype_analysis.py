
"""
SIMCHECK: Failure Subtype Analysis

Compare the types of critical failures discovered
by all five testing strategies at an equal budget.
"""

import pandas as pd
import matplotlib.pyplot as plt


# STEP 1: Load existing experimental data.

pool = pd.read_csv("datasets/candidate_pool.csv")

rankings = pd.read_csv(
    "experiments/frozen_rankings.csv"
)

MODELS = ["random_forest", "neural_network"]

METHODS = [
    "random",
    "uncertainty",
    "estimated_error",
    "action_sensitivity",
    "raas"
]

BUDGET = 300

STATES_ALLOWED = BUDGET // 3

results = []


# STEP 2: Analyze each world model.

for model in MODELS:

    decisions = pd.read_csv(
        f"experiments/{model}_oracle_decisions.csv"
    )

    data = pool.merge(
        decisions,
        on="state_id",
        validate="one_to_one"
    )

    critical = data[data["critical_failure"]]


    # STEP 3: Define the failure subtypes.
    #
    # Avoidable deaths can overlap with the
    # basic attack and skillshot categories.

    subtypes = {
        "basic_attack": (
            critical["enemy_attack"] == "basic"
        ),
        "skillshot": (
            critical["enemy_attack"] == "skillshot"
        ),
        "no_incoming_attack": (
            critical["enemy_attack"] == "none"
        ),
        "avoidable_death": (
            critical["avoidable_death"]
        ),
        "nonfatal_critical": (
            ~critical["avoidable_death"]
        )
    }

    subtype_ids = {
        name: set(critical.loc[mask, "state_id"])
        for name, mask in subtypes.items()
    }


    # STEP 4: Evaluate each testing strategy.

    for method in METHODS:

        selected = rankings[
            (rankings["model"] == model)
            & (rankings["method"] == method)
            & (rankings["rank"] <= STATES_ALLOWED)
        ]

        # Random testing has 100 repetitions.
        # Each other method has one fixed ranking.

        for repetition, group in selected.groupby(
            "repetition"
        ):

            tested_ids = set(group["state_id"])

            assert len(tested_ids) == STATES_ALLOWED

            for subtype, failure_ids in subtype_ids.items():

                discovered = len(
                    tested_ids & failure_ids
                )

                total = len(failure_ids)

                results.append({
                    "model": model,
                    "method": method,
                    "repetition": repetition,
                    "subtype": subtype,
                    "total_failures": total,
                    "discovered": discovered,
                    "coverage_pct": (
                        discovered / total * 100
                        if total > 0 else float("nan")
                    )
                })


# STEP 5: Summarize results across repetitions.

trials = pd.DataFrame(results)

summary = trials.groupby([
    "model",
    "method",
    "subtype"
]).agg(
    total_failures=("total_failures", "first"),
    discovered=("discovered", "mean"),
    coverage_pct=("coverage_pct", "mean")
).reset_index()

trials.to_csv(
    "results/failure_subtype_trials.csv",
    index=False
)

summary.to_csv(
    "results/failure_subtype_summary.csv",
    index=False
)


# STEP 6: Display results and create graphs.

for model in MODELS:

    subset = summary[summary["model"] == model]

    print("\n" + "=" * 60)
    print(model.upper())
    print("=" * 60)

    print("\nTotal failures by subtype:")

    print(
        subset.groupby("subtype")["total_failures"]
        .first().to_string()
    )

    discoveries = subset.pivot(
        index="method",
        columns="subtype",
        values="discovered"
    )

    coverage = subset.pivot(
        index="method",
        columns="subtype",
        values="coverage_pct"
    )

    print("\nMean failures discovered:")
    print(discoveries.round(2).to_string())

    print("\nFailure subtype coverage (%):")
    print(coverage.round(2).to_string())


    # Plot attack-related failure discoveries.
    # Avoidable deaths are omitted from this graph
    # because they overlap with attack categories.

    plot_data = discoveries[[
        "basic_attack",
        "skillshot",
        "no_incoming_attack"
    ]].reindex(METHODS)

    ax = plot_data.plot(
        kind="bar",
        stacked=True,
        figsize=(10, 6),
        color=["#538DCA", "#D89B55", "#999999"]
    )

    ax.set_title(
        f"Failure Subtypes: {model.replace('_', ' ').title()}"
    )

    ax.set_xlabel("Testing Strategy")
    ax.set_ylabel("Critical Failures Discovered")

    plt.xticks(rotation=20)
    plt.tight_layout()

    plt.savefig(
        f"results/{model}_failure_subtypes.png",
        dpi=300
    )

    plt.close()


print("\nAdditional simulator queries used: 0")
print("Failure subtype analysis complete.")

