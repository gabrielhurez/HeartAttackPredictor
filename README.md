# Health Condition Risk Prediction

A machine learning project that predicts a person's risk of 10 chronic health conditions
from survey answers about their health, lifestyle and demographics. It uses the CDC's 2022
BRFSS heart disease survey data (`heart_2022_no_nans.csv`).

Conditions covered: heart attack, angina, stroke, asthma, skin cancer, COPD, depression,
kidney disease, arthritis and diabetes.

## What's in here

- **Notebooks**: data exploration (`EDA.ipynb`, `Descriptive_Stats.ipynb`), cleaning and
  encoding (`Preprocessing.ipynb`), and model training and comparison (`Modeling.ipynb`,
  `Models.ipynb`, `Condition_Models.ipynb`).
- **Models**: `models/` holds two calibrated models per condition, so their outputs are
  real probabilities. A *full* model uses every survey answer, and a *lifestyle* model uses
  only things a person can change or easily report.
- **App**: `app/` is a small web app (FastAPI backend, React frontend) where you answer the
  survey questions and see your estimated risk for each condition compared with the average
  adult. See [app/README.md](app/README.md) for how to run it.

## My part

I built the modeling side of the project:

- Set up the modeling workflow: stratified cross-validation, a shared evaluation helper,
  and MLflow experiment tracking.
- Trained baselines (a dummy classifier and logistic regression), then tree models
  (random forest and histogram gradient boosting).
- Tuned hyperparameters with randomized search and handled class imbalance with class
  weights.
- Evaluated the models with ROC-AUC, PR-AUC, recall and confusion matrices, and chose
  decision thresholds that favor recall.
- Calibrated the models with isotonic regression, so the app shows real probabilities
  instead of raw scores.

**Tech:** Python, pandas, NumPy, scikit-learn, MLflow, matplotlib, Jupyter, joblib.
The team's app uses FastAPI and React.

## Credits

Made August–October 2026, by team KJGJ:
Kay Dickson, Jake Dorick, Gabriel Hurez-Soler and Justin Gauthier.

The original repo is
https://github.com/jdorick/CS4342-Project.
