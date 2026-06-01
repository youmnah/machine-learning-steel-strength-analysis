# Steel Strength Prediction Using Machine Learning

## Overview

This project predicts the mechanical properties of steel alloys using machine learning regression algorithms.

The models use the chemical composition of steel samples (Carbon, Manganese, Silicon, Chromium, Nickel, Molybdenum, etc.) to predict:

* Yield Strength
* Tensile Strength

The project performs data preprocessing, feature scaling, model training, evaluation, and comparison across multiple machine learning algorithms.

## Dataset

The dataset contains steel alloy compositions and corresponding mechanical properties.

### Features

* Carbon (C)
* Manganese (Mn)
* Silicon (Si)
* Chromium (Cr)
* Nickel (Ni)
* Molybdenum (Mo)
* Vanadium (V)
* Nitrogen (N)
* Niobium (Nb)
* Cobalt (Co)
* Tungsten (W)
* Aluminum (Al)
* Titanium (Ti)

### Target Variables

* Yield Strength
* Tensile Strength

## Data Preprocessing

The following preprocessing steps are applied:

1. Remove the `formula` column.
2. Remove rows with missing values in:

   * Carbon (C)
   * Elongation
3. Remove rows containing zero values in:

   * Carbon (C)
   * Elongation
4. Standardize numerical features using StandardScaler.

## Machine Learning Models

The project evaluates the following regression algorithms:

* K-Nearest Neighbors Regressor (KNN)
* Support Vector Regressor (SVR)
* Random Forest Regressor
* Linear Regression
* Decision Tree Regressor
* AdaBoost Regressor
* Gradient Boosting Regressor
* Voting Regressor Ensemble

## Evaluation Metrics

Models are evaluated using:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R² Score

## Visualizations

The notebook includes:

* Missing value analysis
* Correlation heatmaps
* Scatter plot matrices
* Outlier detection using IQR
* Model comparison charts
* RMSE vs R² comparison plots

## Project Structure

.
├── steel_strength_prediction.py
├── requirements.txt
├── README.md
└── .gitignore

## Installation

Clone the repository:

git clone https://github.com/yourusername/steel-strength-prediction.git

Install dependencies:

pip install -r requirements.txt

## Run

python steel_strength_prediction.py

## Results

The project compares multiple regression algorithms and identifies the best-performing model for predicting steel mechanical properties.

## Future Improvements

* Hyperparameter tuning
* Cross-validation
* Feature selection
* XGBoost and LightGBM models
* Model deployment as a web API
* Interactive dashboard for predictions

## Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Plotly
* Missingno
