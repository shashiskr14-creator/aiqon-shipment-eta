
import pandas as pd
from pathlib import Path

# Find and load the dataset
project_dir = Path(__file__).resolve().parent.parent
data_path = project_dir / "data" / "shipments.csv"

df = pd.read_csv(data_path)

print("=== DATASET OVERVIEW ===")
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

print("\n=== COLUMN NAMES ===")
print(df.columns.tolist())

print("\n=== MISSING VALUES ===")
print(df.isnull().sum())

print("\n=== DELAY SUMMARY ===")
print(df[
    [
        "departure_delay_hours",
        "port_delay_hours",
        "customs_delay_hours",
        "final_delay_hours",
    ]
].describe().round(2))

print("\n=== SHIPMENTS BY ROUTE ===")
print(
    df.groupby(["origin", "destination"])
    .size()
    .sort_values(ascending=False)
)

# Check the relationship between factors and final delay
print("\n=== CORRELATION WITH FINAL DELAY ===")
print(
    df[
        [
            "congestion",
            "weather_severity",
            "departure_delay_hours",
            "port_delay_hours",
            "customs_delay_hours",
            "final_delay_hours",
        ]
    ]
    .corr()["final_delay_hours"]
    .sort_values(ascending=False)
    .round(3)
)
