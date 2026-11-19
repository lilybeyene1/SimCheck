
"""
SIMCHECK: Failure Pattern Analysis

Investigate the combat conditions associated
with critical failures and determine which
failures are uniquely discovered by RAAS.
"""

import pandas as pd


# STEP 1: Load our existing experimental data.

pool = pd.read_csv("datasets/candidate_pool.csv")

rankings = pd.read_csv(
    "experiments/frozen_rankings.csv"
)

MODELS = ["random_forest", "neural_network"]

METHODS = [
    "uncertainty",
    "estimated_error",
    "action_sensitivity",
    "raas"
]


# STEP 2: Analyze each world model separately.

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

    discoveries = {}


    # STEP 3: Identify the critical failures
    # discovered by each testing strategy.

    for method in METHODS:

        selected = rankings[
            (rankings["model"] == model)
            & (rankings["method"] == method)
            & (rankings["rank"] <= 100)
        ]

        discoveries[method] = set(
            critical[
                critical["state_id"].isin(
                    selected["state_id"]
                )
            ]["state_id"]
        )


    # STEP 4: Identify failures unique to RAAS.

    competing_failures = set().union(*[
        discoveries[method]
        for method in METHODS
        if method != "raas"
    ])

    unique_raas = (
        discoveries["raas"] - competing_failures
    )

    raas_failures = critical[
        critical["state_id"].isin(discoveries["raas"])
    ]

    unique_failures = critical[
        critical["state_id"].isin(unique_raas)
    ]


    # STEP 5: Define our comparison groups.

    groups = {
        "all_states": data,
        "all_critical_failures": critical,
        "raas_failures": raas_failures,
        "unique_raas_failures": unique_failures
    }

    comparison = []


    # STEP 6: Describe the combat conditions
    # associated with each group.

    for name, group in groups.items():

        comparison.append({
            "group": name,
            "states": len(group),
            "mean_player_hp": group["player_hp"].mean(),
            "mean_enemy_hp": group["enemy_hp"].mean(),
            "mean_distance": group["distance"].mean(),
            "skillshot_pct": (
                group["enemy_attack"] == "skillshot"
            ).mean() * 100,
            "basic_attack_pct": (
                group["enemy_attack"] == "basic"
            ).mean() * 100,
            "dodge_available_pct": (
                group["dodge_cooldown"] == 0
            ).mean() * 100,
            "model_attack_pct": (
                group["model_action"] == "attack"
            ).mean() * 100,
            "avoidable_death_pct": (
                group["avoidable_death"].mean() * 100
            ),
            "mean_regret": group["regret"].mean()
        })

    summary = pd.DataFrame(comparison)


    # STEP 7: Save and display our findings.

    summary.to_csv(
        f"results/{model}_failure_patterns.csv",
        index=False
    )

    unique_failures.to_csv(
        f"results/{model}_unique_raas_failures.csv",
        index=False
    )

    print("\n" + "=" * 60)
    print(model.upper())
    print("=" * 60)

    print("\nFailure pattern comparison:")
    print(summary.round(2).T.to_string())

    print("\nUnique RAAS failure cases:")

    columns = [
        "state_id",
        "player_hp",
        "enemy_hp",
        "distance",
        "enemy_attack",
        "dodge_cooldown",
        "model_action",
        "optimal_action",
        "regret",
        "avoidable_death"
    ]

    print(
        unique_failures[columns]
        .sort_values("regret", ascending=False)
        .to_string(index=False)
    )


print("\nAdditional simulator queries used: 0")
print("Failure pattern analysis complete.")
