
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "shipments.csv"
MODEL_DIR = PROJECT_DIR / "models"
REPORT_DIR = PROJECT_DIR / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

df["scheduled_departure"] = pd.to_datetime(
    df["scheduled_departure"]
)
df["departure_month"] = df["scheduled_departure"].dt.month

TARGET = "final_delay_hours"

# Features available at each prediction checkpoint.
# Never use delays from a stage that has not happened yet.
checkpoints = {
    "before_departure": {
        "numeric": [
            "planned_transit_hours",
            "congestion",
            "weather_severity",
            "departure_month",
        ],
        "categorical": ["origin", "destination"],
    },
    "after_departure": {
        "numeric": [
            "planned_transit_hours",
            "congestion",
            "weather_severity",
            "departure_month",
            "departure_delay_hours",
        ],
        "categorical": ["origin", "destination"],
    },
    "after_port_arrival": {
        "numeric": [
            "planned_transit_hours",
            "congestion",
            "weather_severity",
            "departure_month",
            "departure_delay_hours",
            "port_delay_hours",
        ],
        "categorical": ["origin", "destination"],
    },
    "after_customs": {
        "numeric": [
            "planned_transit_hours",
            "congestion",
            "weather_severity",
            "departure_month",
            "departure_delay_hours",
            "port_delay_hours",
            "customs_delay_hours",
        ],
        "categorical": ["origin", "destination"],
    },
}

# Use the same train/test rows at every checkpoint.
train_indices, test_indices = train_test_split(
    df.index,
    test_size=0.20,
    random_state=42,
)

y_train = df.loc[train_indices, TARGET]
y_test = df.loc[test_indices, TARGET]

all_models = {}
metric_records = []

for checkpoint, config in checkpoints.items():
    print(f"\n=== {checkpoint.upper()} ===")

    numeric_features = config["numeric"]
    categorical_features = config["categorical"]
    features = numeric_features + categorical_features

    X_train = df.loc[train_indices, features]
    X_test = df.loc[test_indices, features]

    preprocessor = ColumnTransformer([
        (
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numeric_features,
        ),
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical_features,
        ),
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=3,
            random_state=42,
        )),
    ])

    baseline = DummyRegressor(strategy="mean")
    baseline.fit(X_train, y_train)
    baseline_predictions = baseline.predict(X_test)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    r2 = r2_score(y_test, predictions)

    baseline_mae = mean_absolute_error(
        y_test, baseline_predictions
    )

    print(f"Baseline MAE: {baseline_mae:.2f} hours")
    print(f"Model MAE:    {mae:.2f} hours")
    print(f"Model RMSE:   {rmse:.2f} hours")
    print(f"Model R2:     {r2:.3f}")

    metric_records.append({
        "checkpoint": checkpoint,
        "baseline_mae_hours": baseline_mae,
        "model_mae_hours": mae,
        "model_rmse_hours": rmse,
        "model_r2": r2,
    })

    all_models[checkpoint] = {
        "model": model,
        "features": features,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
    }

    # Save test predictions for later analysis.
    results = pd.DataFrame({
        "shipment_id": df.loc[test_indices, "shipment_id"].to_numpy(),
        "actual_delay_hours": y_test.to_numpy(),
        "predicted_delay_hours": predictions,
        "baseline_prediction_hours": baseline_predictions,
    })

    results.to_csv(
        REPORT_DIR / f"{checkpoint}_predictions.csv",
        index=False,
    )

    # Save feature importance for this checkpoint.
    names = (
        model.named_steps["preprocessor"]
        .get_feature_names_out()
    )
    importances = (
        model.named_steps["model"].feature_importances_
    )

    importance_df = pd.DataFrame({
        "feature": names,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    importance_df.to_csv(
        REPORT_DIR / f"{checkpoint}_importance.csv",
        index=False,
    )

# Save all stage models and their feature definitions.
joblib.dump(
    all_models,
    MODEL_DIR / "stage_models.joblib",
)

metrics_df = pd.DataFrame(metric_records)
metrics_df.to_csv(
    REPORT_DIR / "stage_metrics.csv",
    index=False,
)

print("\n=== CHECKPOINT COMPARISON ===")
print(metrics_df.round(3).to_string(index=False))

print("\nSaved stage models to models/stage_models.joblib")
print("Saved checkpoint metrics and predictions to reports/")
