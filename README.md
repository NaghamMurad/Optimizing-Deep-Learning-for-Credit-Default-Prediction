# Optimizing-Deep-Learning-for-Credit-Default-Prediction
A Comparative Study of Random Search and Bayesian Optimization

An end-to-end Machine Learning and Deep Learning pipeline designed to predict credit card default using financial tabular data. This repository contrasts traditional baseline models against an optimized deep learning architecture using **Optuna (Bayesian Optimization)** and interprets model decisions using **SHAP (SHapley Additive exPlanations)**.

---

## Key Features

* **Data Preprocessing & Pipeline:** Standardized scaling applied to numerical features with stratified split preservation for class distribution.
* **Custom PyTorch Architecture:** Multi-Layer Perceptron (MLP) built in PyTorch (`CreditNet`) with ReLU activations, Dropout regularization, and Sigmoid output layer.
* **Scikit-Learn Wrapper:** Custom estimator class (`SklearnCreditNet`) enabling seamless integration between PyTorch models and Scikit-Learn/Optuna cross-validation workflows.
* **Hyperparameter Optimization Comparison:**
  * **Baseline:** `RandomizedSearchCV` across hidden dimensions, learning rates, and dropout probabilities.
  * **Advanced:** **Optuna (Bayesian Optimization)** for efficient parameter space exploration.
* **Model Explainability (XAI):**
  * **Global Explainability:** SHAP Summary plots evaluating feature impact across test datasets.
  * **Local Explainability:** SHAP Waterfall plots analyzing individual predictions for financial risk assessment.

---

## Results & Efficiency Comparison

| Optimization Method | Search Time (s) | Best Accuracy / Score |
| :--- | :--- | :--- |
| **Random Search (Baseline)** | ~251.42s | ~82.13% |
| **Bayesian Optimization (Optuna)** | **~184.82s** | **~82.10%** |

*Bayesian Optimization via Optuna achieved comparable performance to Random Search while reducing total tuning runtime by over 26%.*

---

## Model Evaluation & Explainability

### Baseline vs. Deep Learning Performance

* **Logistic Regression Baseline:** Achieved ~81% overall accuracy, but struggled with recall on the minority default class (~24%).
* **Optimized PyTorch Model:**
  * **Accuracy:** ~81.77%
  * **F1 Score:** ~0.49
  * **ROC-AUC:** ~0.67

### Explainability Highlights (SHAP)
* **Global Importance:** Key features influencing credit default predictions include repayment status history (`PAY_0`, `PAY_2`, etc.) and historical bill statement amounts.
* **Local Attribution:** SHAP Waterfall plots allow financial institutions to interpret why a specific user was flagged as high or low default risk, providing model transparency.

---

## Installation & Requirements

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
   cd your-repo-name
