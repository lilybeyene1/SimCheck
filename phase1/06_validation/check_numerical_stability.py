
"""
SIMCHECK: Numerical Stability Investigation

Purpose:
Investigate whether matrix multiplication warnings
originate from our model or numerical libraries.

This script does not modify or retrain our models.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import scipy
import joblib


# Find the main Phase 1 folder
PHASE_ROOT = Path(__file__).resolve().parent.parent

# Point to our existing folders
MODELS = PHASE_ROOT / "models"
DATASETS = PHASE_ROOT / "datasets"


# STEP 1: Identify our software versions

print("\nSOFTWARE ENVIRONMENT")
print("Python:", sys.version.split()[0])
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("Scikit-learn:", sklearn.__version__)
print("SciPy:", scipy.__version__)


# STEP 2: Identify the numerical libraries
# responsible for mathematical operations

print("\nNUMPY CONFIGURATION")
np.show_config()


# STEP 3: Test matrix multiplication using
# ordinary, finite numbers

print("\nMATRIX MULTIPLICATION TEST")

rng = np.random.default_rng(42)

a = rng.normal(size=(1000, 64))
b = rng.normal(size=(64, 64))

with warnings.catch_warnings(record=True) as caught:

    warnings.simplefilter("always")

    result = a @ b

print("Warnings detected:", len(caught))
print("All results finite:", np.isfinite(result).all())

for warning in caught:
    print(warning.message)


# STEP 4: Compare two different methods
# of calculating the same matrix product

reference = np.einsum(
    "ij,jk->ik", a, b, optimize=False
)

print("\nMATRIX ACCURACY CHECK")

print(
    "Results agree:",
    np.allclose(result, reference)
)

print(
    "Maximum numerical difference:",
    np.max(np.abs(result - reference))
)


# STEP 5: Check the existing neural network

saved = joblib.load(MODELS / "neural_network.joblib")

model = saved["model"]

data = pd.read_csv(DATASETS / "validation.csv")

encoded = pd.get_dummies(
    data,
    columns=["enemy_attack", "action"],
    dtype=int
)

X = encoded.reindex(
    columns=saved["features"],
    fill_value=0
)


# STEP 6: Check whether repeated predictions
# produce consistent results

with warnings.catch_warnings(record=True) as caught:

    warnings.simplefilter("always")

    first = model.predict(X)
    second = model.predict(X)

print("\nNEURAL NETWORK STABILITY")

print("Warnings detected:", len(caught))
print("All predictions finite:", np.isfinite(first).all())
print("Repeated predictions agree:", np.allclose(first, second))
print("Maximum absolute prediction:", np.abs(first).max())

for message in sorted(set(str(w.message) for w in caught)):
    print(message)
