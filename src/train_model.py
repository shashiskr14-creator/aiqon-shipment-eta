
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


# --------------------------------------------------
# 1. Set up project paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "shipments.csv"
MODEL_DIR = PROJECT_DIR / "models"
REPORT_DIR = PROJECT_DIR / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# 2. Load the dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

# Extract the departure month from the timestamp
df["scheduled_departure"] = pd.to_datetime(
    df["scheduled_departure"]
)
df["departure_month"] = (
    df["scheduled_departure"].dt.month
)

# The value we want to predict
target = "final_delay_hours"

# Only use information available before departure.
# Do NOT include actual arrival or observed stage delays.
numeric_features = [
    "planned_transit_hours",
    "congestion",
    "weather_severity",
    "departure_month",
]

categorical_features = [
    "origin",
    "destination",
]

features = numeric_features + categorical_features

X = df[features]
y = df[target]

print("Dataset loaded successfully.")
print("Number of shipments:", len(df))
print("Prediction features:", features)


# --------------------------------------------------
# 3. Split training and test data
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)

print("\nTraining shipments:", len(X_train))
print("Testing shipments:", len(X_test))


# --------------------------------------------------
# 4. Prepare the features
# --------------------------------------------------

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])


# --------------------------------------------------
# 5. Create two models
# --------------------------------------------------

baseline = DummyRegressor(strategy="mean")

random_forest = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=3,
        random_state=42,
    )),
])


# --------------------------------------------------
# 6. Evaluate predictions
# --------------------------------------------------

def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    r2 = r2_score(y_test, predictions)

    print(f"\n{name}")
    print("-" * len(name))
    print(f"MAE:  {mae:.2f} hours")
    print(f"RMSE: {rmse:.2f} hours")
    print(f"R2:   {r2:.3f}")

    return {
        "MAE_hours": float(mae),
        "RMSE_hours": rmse,
        "R2": float(r2),
    }, predictions


baseline_metrics, baseline_predictions = evaluate_model(
    "Baseline model",
    baseline,
    X_train, y_train,
    X_test, y_test,
)

rf_metrics, rf_predictions = evaluate_model(
    "Random Forest model",
    random_forest,
    X_train, y_train,
    X_test, y_test,
)


# --------------------------------------------------
# 7. Save the trained model and predictions
# --------------------------------------------------

model_path = MODEL_DIR / "eta_model.joblib"
joblib.dump(random_forest, model_path)

results = X_test.copy()
results["actual_delay_hours"] = y_test
results["baseline_prediction_hours"] = baseline_predictions
results["predicted_delay_hours"] = rf_predictions

# Reconstruct the scheduled ETA for each test shipment
scheduled_arrival = pd.to_datetime(
    df.loc[X_test.index, "scheduled_arrival"]
)
results["predicted_arrival"] = (
    scheduled_arrival
    + pd.to_timedelta(rf_predictions, unit="h")
)

results["actual_arrival"] = pd.to_datetime(
    df.loc[X_test.index, "actual_arrival"]
).to_numpy()

results = results.sort_index()
results.to_csv(REPORT_DIR / "test_predictions.csv", index=False)

metrics = {
    "baseline": baseline_metrics,
    "random_forest": rf_metrics,
    "training_shipments": len(X_train),
    "testing_shipments": len(X_test),
    "target": target,
    "prediction_timing": "Before departure",
}

with open(REPORT_DIR / "metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)


# --------------------------------------------------
# 8. Explain feature importance
# --------------------------------------------------

feature_names = (
    random_forest
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

importances = (
    random_forest
    .named_steps["model"]
    .feature_importances_
)

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances,
}).sort_values("importance", ascending=False)

importance_df.to_csv(
    REPORT_DIR / "feature_importance.csv",
    index=False,
)

print("\nTop 10 model features:")
print(importance_df.head(10).to_string(index=False))

print("\nSaved trained model:", model_path)
print("Saved evaluation results in:", REPORT_DIR)
print("\nTraining and evaluation completed!")
