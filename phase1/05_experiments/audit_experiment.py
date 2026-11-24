
"""
SIMCHECK: Experimental Results Integrity Audit

Verify our experimental results before
performing statistical analysis.
"""

import hashlib
import numpy as np
import pandas as pd
from pathlib import Path


# Find the main Phase 1 folder
PHASE_ROOT = Path(__file__).resolve().parent.parent

# Point to our existing folders
EXPERIMENTS = PHASE_ROOT / "experiments"
DATASETS = PHASE_ROOT / "datasets"
RESULTS = PHASE_ROOT / "results"


# STEP 1: Verify that our frozen rankings
# have not been modified

with open(EXPERIMENTS / "frozen_rankings.csv", "rb") as file:
    fingerprint = hashlib.sha256(file.read()).hexdigest()

with open(EXPERIMENTS / "rankings_sha256.txt") as file:
    expected = file.read().strip()

assert fingerprint == expected

rankings = pd.read_csv(EXPERIMENTS / "frozen_rankings.csv")

pool = pd.read_csv(DATASETS / "candidate_pool.csv")

truth = pd.read_csv(EXPERIMENTS / "oracle_outcomes.csv")

trials = pd.read_csv(RESULTS / "budget_experiment_trials.csv")


# STEP 2: Verify dataset integrity

assert len(pool) == 5000
assert pool["state_id"].is_unique

assert len(truth) == 15000

assert not truth.duplicated(
    ["state_id", "action"]
).any()

assert (truth.groupby("state_id").size() == 3).all()

assert len(rankings) == 20800
assert len(trials) == 832

assert rankings["state_id"].isin(pool["state_id"]).all()

assert not rankings.duplicated([
    "model", "method", "repetition", "state_id"
]).any()

print("SIMCHECK: EXPERIMENTAL INTEGRITY AUDIT")
print("\nFrozen rankings: VERIFIED")
print("Candidate pool: VERIFIED")
print("Oracle outcomes: VERIFIED")
print("Experimental trial count: VERIFIED")


# STEP 3: Audit each world model

for model in ["random_forest", "neural_network"]:

    decisions = pd.read_csv(
        EXPERIMENTS / f"{model}_oracle_decisions.csv"
    )

    assert len(decisions) == 5000
    assert decisions["state_id"].is_unique

    # Independently recalculate decision regret
    # using the oracle outcomes

    optimal = truth.groupby(
        "state_id"
    )["actual_reward"].max()

    chosen = decisions.merge(
        truth,
        left_on=["state_id", "model_action"],
        right_on=["state_id", "action"],
        validate="one_to_one"
    )

    recalculated_regret = (
        chosen["state_id"].map(optimal)
        - chosen["actual_reward"]
    )

    assert np.allclose(
        recalculated_regret,
        chosen["regret"]
    )

    assert (
        (chosen["regret"] >= 20)
        == chosen["critical_failure"]
    ).all()

    # Identify whether survival was possible

    survival_possible = (
        truth.groupby("state_id")[
            "actual_next_player_hp"
        ].max() > 0
    )

    recalculated_death = (
        (chosen["actual_next_player_hp"] == 0)
        & chosen["state_id"].map(survival_possible)
    )

    assert (
        recalculated_death
        == chosen["avoidable_death"]
    ).all()


    # STEP 4: Examine the relationship between
    # critical failures and avoidable deaths

    print("\n" + "=" * 60)
    print(model.upper())
    print("=" * 60)

    print("\nDecision regret: VERIFIED")
    print("Critical failure labels: VERIFIED")
    print("Avoidable death labels: VERIFIED")

    print("\nCritical failure vs. avoidable death:")

    print(pd.crosstab(
        decisions["critical_failure"],
        decisions["avoidable_death"]
    ))


    # STEP 5: Examine failure discovery overlap

    discovered = {}

    methods = [
        "uncertainty",
        "estimated_error",
        "action_sensitivity",
        "raas"
    ]

    for method in methods:

        selected = rankings[
            (rankings["model"] == model)
            & (rankings["method"] == method)
            & (rankings["rank"] <= 100)
        ]

        outcomes = selected.merge(
            decisions,
            on="state_id",
            validate="one_to_one"
        )

        failures = outcomes[
            outcomes["critical_failure"]
        ]

        discovered[method] = set(
            failures["state_id"]
        )

    overlap = pd.DataFrame(
        index=methods,
        columns=methods,
        dtype=int
    )

    for first in methods:
        for second in methods:

            overlap.loc[first, second] = len(
                discovered[first] & discovered[second]
            )

    print("\nShared critical failures at 300 queries:")
    print(overlap.astype(int).to_string())


    # STEP 6: Identify failures uniquely discovered
    # by each testing strategy

    print("\nUnique critical failures by method:")

    for method in methods:

        others = set().union(*[
            discovered[m]
            for m in methods
            if m != method
        ])

        unique = discovered[method] - others

        print(method, ":", len(unique))


    # STEP 7: Save detailed critical failure cases

    failure_cases = decisions[
        decisions["critical_failure"]
    ].merge(pool, on="state_id")

    failure_cases.to_csv(
        RESULTS / f"{model}_critical_failure_cases.csv",
        index=False
    )

    overlap.to_csv(
        RESULTS / f"{model}_failure_overlap.csv"
    )


print("\nAll implemented integrity checks passed.")
print("Additional simulator queries used: 0")
