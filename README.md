# Optimizing Deep Learning for Credit Default Prediction

### A Comparative Study of Random Search and Bayesian Optimization

An end-to-end machine learning and deep learning project for predicting credit card default using financial tabular data. This project compares a traditional Logistic Regression baseline with a custom PyTorch neural network and investigates two hyperparameter optimization approaches: **RandomizedSearchCV** and **Optuna Bayesian Optimization**. Model predictions are also interpreted using **SHAP (SHapley Additive exPlanations)**.

## Key Features

* **Data Preprocessing:** Standardized scaling of numerical features with stratified splitting to preserve class distribution.
* **Custom PyTorch Model:** Multi-Layer Perceptron (`CreditNet`) using ReLU activations, Dropout regularization, and a Sigmoid output layer.
* **Scikit-Learn Integration:** Custom `SklearnCreditNet` estimator for integrating the PyTorch model with Scikit-Learn workflows.
* **Hyperparameter Optimization:**

  * `RandomizedSearchCV` as a baseline search method.
  * Optuna Bayesian Optimization for guided hyperparameter search.
* **Model Explainability:**

  * SHAP summary plots for global feature importance.
  * SHAP waterfall plots for individual predictions.

## Results

| Optimization Method            | Search Time | Best Accuracy / Score |
| ------------------------------ | ----------: | --------------------: |
| Random Search                  |   ~251.42 s |               ~82.13% |
| Bayesian Optimization (Optuna) |   ~184.82 s |               ~82.10% |

Optuna achieved a similar best score to Random Search while reducing the reported tuning time by more than 26%.

## Model Evaluation

The Logistic Regression baseline achieved approximately **81% overall accuracy**, with approximately **24% recall for the minority default class**.

The optimized PyTorch model achieved:

| Metric   |  Result |
| -------- | ------: |
| Accuracy | ~81.77% |
| F1 Score |   ~0.49 |
| ROC-AUC  |   ~0.67 |

These results illustrate why accuracy alone can be misleading for an imbalanced classification problem. F1-score, recall, and ROC-AUC provide additional information about model performance.

## Model Explainability

SHAP was used to examine which features contributed to the model's predictions.

The analysis identified repayment-history variables such as `PAY_0` and `PAY_2`, along with historical bill statement amounts, among the important features influencing predictions.

SHAP waterfall plots were also used to examine how individual features contributed to individual model predictions.

## Technologies

* Python
* PyTorch
* Scikit-Learn
* Optuna
* SHAP
* Pandas
* NumPy
* Matplotlib
* Jupyter Notebook

## Repository Contents

```text
├── Machine_Learning_Optimization_Nagham_Murad.ipynb
├── Presentation ML Optimization Nagham Murad.pdf
├── Report ML Optimization Nagham Murad.pdf
└── README.md
```

## Project Documentation

* [Jupyter Notebook](Machine_Learning_Optimization_Nagham_Murad.ipynb)
* [Project Report](Report%20ML%20Optimization%20Nagham%20Murad.pdf)
* [Project Presentation](Presentation%20ML%20Optimization%20Nagham%20Murad.pdf)
