# LTV Prediction Model with LSTM

Deep learning model for User Lifetime Value (LTV) prediction using LSTM networks. This project implements daily LTV predictions with automated correction based on historical patterns.

## Overview

- LSTM-based LTV prediction
- Growth rate analysis and pattern detection  
- Automated correction system
- Continuous learning from new data

## Prerequisites

```txt
tensorflow>=2.0.0
pandas>=1.0.0
numpy>=1.18.0
scikit-learn>=0.24.0
```

## Installation
```python
git clone https://github.com/yyyopa/ltv_prediction.git
cd ltv_prediction
pip install -r requirements.txt
```

## Usage
```python
from ltv_run_analysis import run_ltv_analysis

results = run_ltv_analysis(
    base_path='path/to/data',
    prediction_days=30,
    epochs=50
)
```
## Data Format
Input files should be CSV format with following columns:

- Date: Weekly dates (format: "YYYY-MM-DD~YYYY-MM-DD")
- day_0 to day_360: Daily LTV values
- Additional metadata columns (Apps, OS, Country, etc.)

## Directory Structure
```
LTV_ML/
├── Files/                  # Original LTV data
├── Files_Predictions/      # Model predictions
└── Files_Predictions_Final/# Corrected predictions
```

## Model Architecture
- LSTM layers (32, 16 units)
- Dropout (0.2)
- Dense layers (8 units, ReLU)
- Output layer (Softplus activation)

## Results
Model Performance (Feb 2025):
* Training MAE: 0.0014 
* Validation MAE: 0.0013
* Average prediction accuracy: 98.5%

Key findings from validation results:
* Short-term prediction (D7-D10):
 - Average accuracy: 99.0%
 - Variance range: ±0.034~0.116
* Mid-term prediction (D20-D30):
 - Average accuracy: 97.7%
 - Consistent positive trend in accuracy
* Long-term prediction (D30-D36):
 - Average accuracy: 95.8%
 - Maximum deviation: 4.83%

Model training characteristics:
* MAE stabilizes around epoch 40
* Final validation loss: 5.09e-6
* Robust performance across 50 epochs
* Successfully processed 123,028 sequences

![Prediction Accuracy](results/prediction_accuracy.png)
![MAE Trend](results/mae_trend.png)

## License
MIT License

## Author
Ilyeop
