# Car Price Prediction

A Machine Learning project that explores a car pricing dataset, performs data preprocessing and feature engineering, and applies Linear Regression to predict car prices.

## Project Overview

The goal of this project was to understand and apply the basic Machine Learning workflow to a car price prediction problem.

The project covers:

* Exploratory Data Analysis (EDA)
* Data Cleaning
* Feature Engineering
* Feature Selection
* Categorical Encoding
* Train-Test Split
* Linear Regression
* Model Evaluation
* Residual Analysis
* R² and Adjusted R²
* Overfitting and Underfitting analysis

## Dataset

The dataset contains information about cars and their prices.

The target variable is:

* `Price`

The features used for the final Linear Regression model were:

* `Engine Size`
* `Mileage_per_Year`
* `Car_Age`
* `Model_GLC`
* `Brand_BMW`
* `Engine_Category_Medium`

> **Note:** These categorical features were identified as statistically significant via Chi-square tests, but given the multiple testing problem (40 tests at α=0.05), they may represent false positives rather than true predictive signals.

## Project Workflow

```text
Dataset
   ↓
Exploratory Data Analysis
   ↓
Data Cleaning
   ↓
Feature Engineering
   ↓
Feature Selection
   ↓
One-Hot Encoding
   ↓
Train-Test Split
   ↓
Linear Regression
   ↓
Model Evaluation
```

## Feature Selection Findings

During feature selection, all numeric features had very weak Pearson correlations with `Price`, with correlation values below `0.05`.

Chi-square tests were also performed on categorical features. Only 3 significant results were found out of 40 tests at α = 0.05, which can be consistent with random chance when many statistical tests are performed.

These findings suggested that the available features had limited useful relationships with the target variable.

## Model

A Linear Regression model was trained using an 80/20 train-test split.

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
```

Dataset split:

* Training samples: `2000`
* Testing samples: `500`

## Model Evaluation

The model produced the following results:

| Metric      |           Result |
| ----------- | ---------------: |
| Training R² |        `0.00646` |
| Testing R²  |       `-0.00199` |
| Adjusted R² |       `-0.01419` |
| Test MSE    | `759,065,554.85` |

### Interpretation

The negative test R² indicates that the Linear Regression model performs slightly worse than simply predicting the average car price for every test example.

The very low training and testing R² values indicate that the model is **underfitting** and is not capturing meaningful patterns in the dataset.

The predictions were also concentrated around the middle price range rather than accurately following the variation in actual car prices.

## Key Learning

This project helped me understand that a Machine Learning model does not automatically perform well just because the model can be trained successfully.

The quality and relationships within the dataset are important. Feature selection and statistical analysis showed that the available features had very weak relationships with `Price`, which was reflected in the final model performance.

## What I Would Do Differently

With a better dataset, I would first verify that the target variable was generated from real-world pricing data rather than synthetically.

I would then:

* Check the relationships between features and the target before extensive feature engineering.
* Investigate whether meaningful non-linear relationships exist.
* Use additional relevant real-world data if available.
* Compare Linear Regression with non-linear models such as Random Forest.
* Evaluate and compare the models using appropriate metrics.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook

## Conclusion

This project was primarily a learning exercise in applying the Machine Learning workflow to a regression problem. Although the Linear Regression model achieved poor performance, the analysis helped identify why the model was not learning useful patterns from the available data and provided practical experience with feature selection and model evaluation.

## Author

**Iqra Hasnain** — Aspiring Data Scientist
[GitHub](https://github.com/IqraHasnain) | [LinkedIn](https://www.linkedin.com/in/iqrahasnain)
