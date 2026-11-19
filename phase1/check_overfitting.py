
"""
SIMCHECK: Overfitting Check

Purpose:
Compare training and testing prediction errors
to investigate whether our world models overfit.

We use our existing trained models without retraining.
"""

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


# STEP 1: Load and encode our original dataset.

data = pd.read_csv("combat_data.csv")

data = pd.get_dummies(
    data,
    columns=["enemy_attack", "action"],
    dtype=int
)


# STEP 2: Separate inputs from prediction targets.

target_columns = [
    "next_player_hp",
    "next_enemy_hp",
    "next_distance",
    "next_dodge_cooldown"
]

X = data.drop(columns=target_columns)
y = data[target_columns]


# STEP 3: Reproduce our original data split.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# STEP 4: Identify our existing trained models.

models = ["random_forest", "neural_network"]


# STEP 5: Evaluate each model without retraining.

for name in models:

    saved = joblib.load(f"models/{name}.joblib")

    model = saved["model"]
    features = saved["features"]

    # Match the original feature order.
    train_features = X_train.reindex(
        columns=features, fill_value=0
    )

    test_features = X_test.reindex(
        columns=features, fill_value=0
    )

    # Generate predictions on both datasets.
    train_predictions = model.predict(train_features)
    test_predictions = model.predict(test_features)

    print("\n" + "=" * 55)
    print("MODEL:", name.upper())
    print("=" * 55)

    # STEP 6: Compare errors for each predicted variable.

    for i, column in enumerate(target_columns):

        train_mae = mean_absolute_error(
            y_train[column],
            train_predictions[:, i]
        )

        test_mae = mean_absolute_error(
            y_test[column],
            test_predictions[:, i]
        )

        # Measure how much the error increases
        # when moving from training to testing.
        gap = test_mae - train_mae

        print(f"\n{column}")
        print(f"Training MAE: {train_mae:.4f}")
        print(f"Testing MAE:  {test_mae:.4f}")
        print(f"Error gap:    {gap:.4f}")
