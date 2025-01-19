# Customer Lifetime Value Prediction using Deep Learning

## Project Overview
This project implements an advanced deep learning model to predict Customer Lifetime Value (CLV/LTV) in digital marketing contexts. Using bidirectional GRU (Gated Recurrent Unit) neural networks, the model analyzes temporal patterns in customer behavior to forecast future value generation.

### Key Features
- Bidirectional GRU architecture for capturing complex temporal dependencies
- Time series analysis with sliding window approach
- Robust data preprocessing and feature engineering
- K-fold cross-validation for model reliability
- Comprehensive evaluation metrics including RMSE, MAE, and R²
- TensorBoard integration for training visualization
- Iterative prediction capability for long-term forecasting

## Technical Architecture

### Data Processing Pipeline
1. **Data Loading & Preprocessing**
   - Dynamic file discovery and sorting
   - Automated train/validation/test splitting (70:15:15)
   - Anomaly detection and handling
   - Feature scaling and normalization

2. **Feature Engineering**
   - Temporal feature extraction
   - Rolling statistics computation
   - Sequential pattern analysis
   - Multi-dimensional feature space creation

3. **Model Architecture**
   - Bidirectional GRU layers
   - Layer normalization
   - Dropout regularization
   - Dense layers for final prediction

### Implementation Details
- **Framework**: PyTorch
- **Key Libraries**: 
  - pandas: Data manipulation
  - numpy: Numerical computations
  - scikit-learn: Metrics and preprocessing
  - tensorboard: Training visualization
  - tqdm: Progress monitoring

## Model Performance
- **Evaluation Metrics**:
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - Mean Absolute Error (MAE)
  - R² Score
- **Visualization**:
  - Training/Validation loss curves
  - Prediction vs Actual plots
  - Residual analysis
  - Feature importance analysis

## Business Applications
1. **Marketing Strategy Optimization**
   - Customer segmentation
   - Campaign budget allocation
   - ROI forecasting
   - Customer retention planning

2. **Financial Planning**
   - Revenue prediction
   - Resource allocation
   - Risk assessment
   - Investment planning

## Future Enhancements
- Implementation of attention mechanisms
- Integration of external market indicators
- Real-time prediction capabilities
- API development for business integration
- Extended feature engineering pipeline

## Installation & Usage

### Prerequisites
```bash
python >= 3.8
pytorch
pandas
numpy
scikit-learn
tensorboard
tqdm
```

### Setup
```bash
git clone [repository-url]
cd ltv-prediction
pip install -r requirements.txt
```

### Training
```bash
python train.py --data_path /path/to/data --epochs 30 --batch_size 32
```

### Prediction
```bash
python predict.py --model_path /path/to/model --data_path /path/to/data
```

## Project Structure
```
ltv-prediction/
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── training.py
│   └── architecture.py
├── utils/
│   ├── preprocessing.py
│   └── evaluation.py
├── notebooks/
│   └── analysis.ipynb
├── tests/
├── requirements.txt
└── README.md
```

## Academic Citations
If you use this project in your research, please cite:
```bibtex
@misc{ltv-prediction,
  author = {[Your Name]},
  title = {Customer Lifetime Value Prediction using Deep Learning},
  year = {2024},
  publisher = {GitHub},
  url = {[repository-url]}
}
```

## License
MIT License
