# Shipment ETA Prediction and Delay Risk Estimation

## 1. Project Overview

This project predicts shipment arrival delays and estimates final arrival times using machine learning. It also identifies important factors associated with delays and evaluates how predictions change as shipment information becomes available.

The dataset is synthetically generated for this assignment and does not represent real shipment tracking data.

## 2. Objectives

- Predict final shipment delay in hours.
- Estimate the predicted arrival timestamp.
- Classify predicted delay into illustrative LOW, MEDIUM, and HIGH categories.
- Identify important features influencing predictions.
- Compare prediction accuracy at four operational checkpoints.

## 3. Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib
- Joblib

## 4. Dataset

The dataset contains 250 synthetic shipment records across five shipping routes.

Important fields include:
- Origin and destination
- Scheduled departure and arrival
- Planned transit duration
- Congestion and weather severity
- Departure, port, and customs delays
- Recovery time
- Final delay and actual arrival timestamp

The target variable is `final_delay_hours`.

## 5. Methodology

1. Generate synthetic shipment data.
2. Explore the dataset and inspect delay patterns.
3. Split data into training and test sets.
4. Train a Random Forest regression model.
5. Train checkpoint-specific models using only the features intended to be available at each checkpoint.
6. Compare model predictions against a mean-prediction baseline.
7. Generate predictions and a checkpoint comparison chart.

The four checkpoints are:
- Before departure
- After departure
- After port arrival
- After customs

## 6. Evaluation Results

On the held-out test set of 50 shipments:

| Checkpoint | Model MAE (hours) | Model RMSE (hours) | R² |
|---|---:|---:|---:|
| Before departure | 5.71 | 7.29 | 0.654 |
| After departure | 5.14 | 6.39 | 0.734 |
| After port arrival | 3.47 | 4.47 | 0.870 |
| After customs | 3.12 | 4.05 | 0.893 |

The mean-prediction baseline MAE was approximately 9.52 hours.

These results are from synthetic data. They demonstrate the workflow, not expected performance on real-world shipments.

## 7. Feature Importance

The exploratory analysis found the strongest correlations with final delay for:
- Port delay: 0.890
- Customs delay: 0.667
- Departure delay: 0.604
- Weather severity: 0.587
- Congestion: 0.560

These are correlations in the synthetic dataset and should not be interpreted as proof of causation.

## 8. How to Run

Activate the virtual environment on macOS:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Generate the dataset:

```bash
python src/generate_data.py
```

Explore the dataset:

```bash
python src/explore_data.py
```

Train the main model:

```bash
python src/train_model.py
```

Train checkpoint-specific models:

```bash
python src/train_stage_models.py
```

Run an interactive prediction:

```bash
python src/predict.py
```

Generate the comparison chart:

```bash
python src/plot_results.py
```

## 9. Limitations and Future Work

- The dataset is synthetic and small.
- Delay-risk categories use illustrative thresholds, not calibrated probabilities.
- The evaluation uses one train/test split.
- The model has not been validated on real operational shipment data.
- Future work could use historical shipment records, time-based validation, prediction intervals, and calibrated delay probabilities.

## 10. Conclusion

The project demonstrates an end-to-end shipment ETA prediction workflow, including synthetic data generation, exploratory analysis, regression modelling, checkpoint-specific prediction, evaluation, and visualization.
