
from pathlib import Path

import joblib
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "shipments.csv"
MODEL_PATH = PROJECT_DIR / "models" / "stage_models.joblib"
REPORT_DIR = PROJECT_DIR / "reports"

# Load dataset and trained models
df = pd.read_csv(DATA_PATH)
models = joblib.load(MODEL_PATH)

CHECKPOINTS = {
    "1": "before_departure",
    "2": "after_departure",
    "3": "after_port_arrival",
    "4": "after_customs",
}


def classify_delay(delay_hours):
    """Return an illustrative delay-risk category."""
    if delay_hours < 12:
        return "LOW"
    elif delay_hours < 24:
        return "MEDIUM"
    else:
        return "HIGH"


def predict_shipment(shipment_id, checkpoint):
    # Find the shipment
    matches = df[df["shipment_id"] == shipment_id]

    if matches.empty:
        print("Shipment ID not found.")
        return

    shipment = matches.iloc[0]
    shipment = shipment.copy()
    shipment["departure_month"] = pd.to_datetime(
    shipment["scheduled_departure"]
).month
    model_info = models[checkpoint]

    # Use only the features available at this checkpoint
    features = model_info["features"]
    X = pd.DataFrame(
        [shipment[features].to_dict()],
        columns=features,
    )

    # Predict final delay in hours
    model = model_info["model"]
    predicted_delay = max(0.0, float(model.predict(X)[0]))

    # Calculate predicted arrival timestamp
    scheduled_arrival = pd.to_datetime(
        shipment["scheduled_arrival"]
    )
    predicted_arrival = (
        scheduled_arrival
        + pd.to_timedelta(predicted_delay, unit="h")
    )

    risk = classify_delay(predicted_delay)

    print("\n========== SHIPMENT ETA PREDICTION ==========")
    print(f"Shipment ID: {shipment_id}")
    print(f"Route: {shipment['origin']} -> {shipment['destination']}")
    print(f"Prediction checkpoint: {checkpoint}")
    print(f"Scheduled arrival: {scheduled_arrival}")
    print(f"Predicted delay: {predicted_delay:.2f} hours")
    print(f"Predicted arrival: {predicted_arrival}")
    print(f"Delay risk: {risk}")

    # Show global feature importance for this checkpoint.
    # These explain overall model behaviour, not this shipment alone.
    importance_path = (
        REPORT_DIR / f"{checkpoint}_importance.csv"
    )

    if importance_path.exists():
        importance = pd.read_csv(importance_path)

        print("\nTop overall model drivers:")
        for _, row in importance.head(3).iterrows():
            print(
                f"- {row['feature']}: "
                f"{row['importance']:.3f}"
            )

    print("\nNote:")
    print("- Risk bands are illustrative thresholds, not probabilities.")
    print("- This demo uses synthetic shipment records.")
    print("- Predictions should not be treated as real shipping ETAs.")


def main():
    print("Aiqon Shipment ETA Prediction Demo")
    print("----------------------------------")
    print("Available shipment IDs: 1 to", len(df))

    shipment_input = input(
        "\nEnter shipment ID (1-250): "
    ).strip()

    try:
        shipment_id = int(shipment_input)
    except ValueError:
        print("Please enter a valid integer shipment ID.")
        return

    print("\nChoose prediction checkpoint:")
    print("1. Before departure")
    print("2. After departure")
    print("3. After port arrival")
    print("4. After customs")

    choice = input("Enter choice (1-4): ").strip()

    if choice not in CHECKPOINTS:
        print("Invalid checkpoint choice.")
        return

    predict_shipment(shipment_id, CHECKPOINTS[choice])


if __name__ == "__main__":
    main()
