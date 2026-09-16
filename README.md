# Turbofan Engine Remaining Useful Life (RUL) Predictor

Predict the remaining useful life of NASA C-MAPSS turbofan engines using sensor data.

**Live Demo:** [Add Streamlit Cloud link here later]

## Project Overview
This project uses the NASA C-MAPSS FD001 dataset to train an XGBoost regression model that estimates Remaining Useful Life (RUL) from the latest sensor readings of a turbofan engine.

### Key Features
- Piecewise-linear RUL labeling (clip at 125 cycles)
- Automatic dropping of constant sensors
- Feature scaling + XGBoost regressor
- Interactive Streamlit web app
- Feature importance visualization

### Model Performance (FD001 test set)
| Metric       | Value   |
|--------------|---------|
| RMSE         | ~18     |
| MAE          | ~13     |
| Concordance  | ~0.82   |

## How to Run Locally

```bash
git clone https://github.com/jack-campbell-datascientist/turbofan-rul-predictor.git
cd turbofan-rul-predictor
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py