# house-prices-kaggle

## 📌 Project Overview

This project is based on the Kaggle competition [House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).

The goal is to predict the final sale price of residential homes based on 80+ features describing the houses.

The competition uses Root Mean Squared Error (RMSE) on the log-transformed target (SalePrice) as the evaluation metric.

## ⚡ Approaches & Results
1.Baseline: Random Forest Regressor

- Applied preprocessing with imputation and one-hot encoding.

- Trained using RandomForestRegressor.

- Public Score: 0.14798 (RMSE).

2.Gradient Boosting: LightGBM

- Used the same preprocessing pipeline.

- Applied log1p transformation on the target variable to handle skewness.

- Trained using LightGBM Regressor with tuned hyperparameters.

- Public Score: 0.12921 (RMSE).

3.Exploratory Data Analysis (EDA)

A separate EDA notebook was created to explore the dataset:

- Checked data structure and missing values.

- Investigated target distribution (original vs log-transformed).

- Analyzed correlations between numerical features and SalePrice.

- Visualized categorical features (e.g., Neighborhood) and their impact on housing prices.

## 📊 Key Learnings

- Log-transforming SalePrice improves performance due to its heavy skew.

- Neighborhood and OverallQual are among the strongest predictors.

- Tree-based models (RF, LGBM) outperform simple linear approaches due to mixed numerical/categorical data.