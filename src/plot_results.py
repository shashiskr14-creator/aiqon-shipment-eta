
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
REPORT_DIR = PROJECT_DIR / "reports"

metrics_path = REPORT_DIR / "stage_metrics.csv"

if not metrics_path.exists():
    print("Error: reports/stage_metrics.csv not found.")
    print("Run: python src/train_stage_models.py")
    raise SystemExit(1)

metrics = pd.read_csv(metrics_path)

labels = [
    "Before departure",
    "After departure",
    "After port arrival",
    "After customs",
]

plt.figure(figsize=(9, 5))

plt.plot(
    labels,
    metrics["model_mae_hours"],
    marker="o",
    label="Random Forest MAE",
)

plt.plot(
    labels,
    metrics["baseline_mae_hours"],
    marker="o",
    label="Baseline MAE",
)

plt.ylabel("Mean Absolute Error (hours)")
plt.xlabel("Prediction checkpoint")
plt.title("Shipment ETA Prediction Error by Checkpoint")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

output_path = REPORT_DIR / "checkpoint_comparison.png"
plt.savefig(output_path, dpi=150)
plt.close()

print("Chart saved to:", output_path)
