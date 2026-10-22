
import numpy as np
import pandas as pd
from pathlib import Path

# Make results reproducible
np.random.seed(42)

# Number of shipments to simulate
N_SHIPMENTS = 250

# Possible shipment routes
routes = [
    ("Shanghai", "Los Angeles"),
    ("Singapore", "Rotterdam"),
    ("Mumbai", "Dubai"),
    ("Chennai", "Singapore"),
    ("Rotterdam", "New York"),
]

records = []

for shipment_id in range(1, N_SHIPMENTS + 1):

    # Select a route
    origin, destination = routes[
        np.random.randint(0, len(routes))
    ]

    # Planned travel duration, in hours
    scheduled_departure = pd.Timestamp("2026-01-01") + pd.Timedelta(
        hours=int(np.random.randint(0, 24 * 90))
    )

    planned_transit_hours = int(np.random.randint(120, 401))

    # External conditions
    congestion = np.random.randint(1, 11)  # 1 to 10
    weather_severity = np.random.randint(0, 6)  # 0 to 5

    # Stage 1: departure delay
    departure_delay = max(
        0,
        np.random.normal(2 + congestion * 0.5, 4)
    )

    # Stage 2: port arrival delay
    port_delay = max(
        0,
        departure_delay * 0.65
        + congestion * 1.2
        + weather_severity * 1.5
        + np.random.normal(0, 4)
    )

    # Stage 3: customs processing delay
    customs_delay = max(
        0,
        congestion * 0.8
        + weather_severity * 0.7
        + np.random.normal(0, 3)
    )

    # Recovery buffer: later stages can recover some time
    recovery_hours = np.random.uniform(0, 5)

    # Final delay is influenced by earlier delays and conditions
    final_delay = max(
        0,
        departure_delay * 0.5
        + port_delay * 0.8
        + customs_delay
        + weather_severity * 1.5
        - recovery_hours
        + np.random.normal(0, 3)
    )

    scheduled_arrival = (
        scheduled_departure
        + pd.Timedelta(hours=planned_transit_hours)
    )

    actual_arrival = scheduled_arrival + pd.Timedelta(
        hours=float(final_delay)
    )

    records.append({
        "shipment_id": shipment_id,
        "origin": origin,
        "destination": destination,
        "scheduled_departure": scheduled_departure,
        "scheduled_arrival": scheduled_arrival,
        "planned_transit_hours": planned_transit_hours,
        "congestion": congestion,
        "weather_severity": weather_severity,
        "departure_delay_hours": round(departure_delay, 2),
        "port_delay_hours": round(port_delay, 2),
        "customs_delay_hours": round(customs_delay, 2),
        "recovery_hours": round(recovery_hours, 2),
        "final_delay_hours": round(final_delay, 2),
        "actual_arrival": actual_arrival,
    })

# Convert records into a DataFrame
df = pd.DataFrame(records)

# Save CSV in the project's data folder
output_path = Path(__file__).resolve().parent.parent / "data" / "shipments.csv"
df.to_csv(output_path, index=False)

print("Synthetic shipment dataset created successfully!")
print(f"Total shipments: {len(df)}")
print(f"Saved to: {output_path}")

print("\nFirst 5 shipments:")
print(df.head())

print("\nFinal delay statistics:")
print(df["final_delay_hours"].describe())
