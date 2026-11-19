
"""
SIMCHECK: Statistical Significance Analysis

Compare intelligent testing strategies against
uniform random sampling using exact
hypergeometric tests.


"""

import numpy as np
import pandas as pd

from scipy.stats import hypergeom


# STEP 1: Load experimental results.

summary = pd.read_csv(
    "results/budget_experiment_summary.csv"
)

MODELS = ["random_forest", "neural_network"]

ALPHA = 0.05

statistical_results = []


# STEP 2: Evaluate each world model.

for model in MODELS:

    truth = pd.read_csv(
        f"experiments/{model}_oracle_decisions.csv"
    )

    # Total population size.

    N = len(truth)

    # Total critical failures in the population.

    K = int(truth["critical_failure"].sum())

    subset = summary[
        (summary["model"] == model)
        & (summary["method"] != "random")
    ]


    # STEP 3: Test each strategy and budget.

    for row in subset.itertuples(index=False):

        # Number of states tested.

        n = int(row.query_budget // 3)

        # Observed critical failures.

        k = int(row.mean_failures)

        # Expected failures under random sampling.

        expected_random = n * K / N

        # Probability of randomly discovering
        # at least k critical failures.

        p_value = hypergeom.sf(k - 1, N, K, n)

        # Observed improvement over the theoretical
        # random expectation.

        enrichment = (
            k / expected_random
            if expected_random > 0
            else np.nan
        )

        statistical_results.append({
            "model": model,
            "method": row.method,
            "query_budget": row.query_budget,
            "states_tested": n,
            "observed_failures": k,
            "expected_random": expected_random,
            "enrichment": enrichment,
            "p_value": p_value
        })


# STEP 4: Correct for multiple hypothesis tests.
#
# Holm correction controls the family-wise
# error rate across all 32 comparisons.

results = pd.DataFrame(statistical_results)

results = results.sort_values("p_value").reset_index(
    drop=True
)

number_of_tests = len(results)

multipliers = np.arange(number_of_tests, 0, -1)

adjusted = results["p_value"] * multipliers

results["adjusted_p_value"] = np.minimum(
    1,
    np.maximum.accumulate(adjusted)
)

results["significant_unadjusted"] = (
    results["p_value"] < ALPHA
)

results["significant_adjusted"] = (
    results["adjusted_p_value"] < ALPHA
)


# STEP 5: Save the statistical results.

results.to_csv(
    "results/statistical_analysis.csv",
    index=False
)


# STEP 6: Display the findings.

print("SIMCHECK: STATISTICAL SIGNIFICANCE ANALYSIS")

print("\nSignificance level:", ALPHA)

print("Total hypothesis tests:", number_of_tests)

print("Multiple-testing correction: Holm")

columns = [
    "method",
    "query_budget",
    "observed_failures",
    "expected_random",
    "enrichment",
    "p_value",
    "adjusted_p_value",
    "significant_adjusted"
]

for model in MODELS:

    print("\n" + "=" * 65)
    print(model.upper())
    print("=" * 65)

    subset = results[
        results["model"] == model
    ].sort_values(["query_budget", "p_value"])

    print(subset[columns].round(6).to_string(index=False))


# STEP 7: Summarize significant comparisons.

significant = results[
    results["significant_adjusted"]
]

print("\nSignificant results before correction:")

print(results["significant_unadjusted"].sum())

print("\nSignificant results after correction:")

print(len(significant))

print("\nAdditional simulator queries used: 0")

print("Statistical analysis complete.")
