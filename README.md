# Credit Risk Probability Model for Alternative Data

An end-to-end, production-ready credit risk modeling system built for a Buy-Now-Pay-Later (BNPL) use case.  
The project transforms eCommerce transaction behavior into credit risk signals, trains and tracks models, and exposes real-time risk scoring through an API.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Business Context](#business-context)
- [Credit Scoring Business Understanding](#credit-scoring-business-understanding)
- [Objectives](#objectives)
- [Dataset Overview](#dataset-overview)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [Modeling Approach](#modeling-approach)
- [MLOps & Engineering Practices](#mlops--engineering-practices)
- [Setup & Installation](#setup--installation)
- [Running the Project](#running-the-project)
- [API Usage](#api-usage)
- [Technologies Used](#technologies-used)

---

## Project Overview

This project implements a **Credit Risk Probability Model** using alternative behavioral data from an eCommerce platform.  
It is designed for a financial institution offering BNPL services, where traditional credit bureau data is unavailable or limited.

The system covers the full lifecycle:

- Data ingestion and cleaning
- Feature engineering (including WoE & IV)
- Proxy default label creation using RFM analysis
- Model training, tuning, and evaluation
- Experiment tracking and model registry (MLflow)
- Automated pipelines (DVC)
- Real-time scoring via FastAPI
- Containerized deployment with Docker

---

## Business Context

Bati Bank is partnering with a fast-growing eCommerce platform to offer BNPL services.  
Before extending credit, the bank must estimate the likelihood that a customer will default.

Unlike traditional lending, no historical loan repayment data exists.  
Instead, customer transaction behavior is used as a **proxy signal for credit risk**.

The outcome of this project is a **risk probability score** that supports:
- Loan approval decisions
- Credit limits
- Loan duration optimization
- Portfolio risk monitoring

---

## Credit Scoring Business Understanding

### Basel II and Model Governance

The Basel II Capital Accord emphasizes accurate risk measurement, transparency, and documentation.  
As a result, credit models must be explainable, traceable, and auditable.

This project addresses those requirements by:
- Using well-defined proxy targets
- Logging all experiments and parameters
- Supporting interpretable techniques such as WoE and Logistic Regression
- Enabling versioned model management via MLflow

---

### Proxy Default Variable

The dataset does not contain a direct `default` label.  
To enable supervised learning, a **proxy target variable** is created using customer engagement behavior.

We compute **Recency, Frequency, and Monetary (RFM)** metrics and cluster customers into behavioral segments.  
The least engaged segment is labeled as **high-risk**.

**Business risk:**  
This proxy may not perfectly represent true default behavior. Predictions should be treated as **risk indicators**, not absolute outcomes, and must be refined as real repayment data becomes available.

---

### Model Complexity vs Interpretability

- **Interpretable models (Logistic Regression + WoE)**  
  - Easier to explain to regulators  
  - Transparent feature impact  
  - Preferred for governance  

- **Complex models (Random Forest, Gradient Boosting)**  
  - Higher predictive power  
  - Capture non-linear patterns  
  - Harder to explain  

This project supports both approaches, allowing performance benchmarking while maintaining regulatory alignment.

---

## Objectives

- Define a proxy credit risk label
- Engineer predictive behavioral features
- Train and compare multiple ML models
- Output a probability of default
- Convert probability into a credit score
- Serve predictions via a production-grade API
- Ensure reproducibility and automation

---

## Dataset Overview

**Source:** eCommerce transaction logs  

Key fields include:

| Column | Description |
|------|-------------|
| TransactionId | Unique transaction identifier |
| CustomerId | Unique customer identifier |
| Amount | Transaction amount |
| ProductCategory | Product category |
| ChannelId | Platform channel |
| TransactionStartTime | Timestamp |
| FraudResult | Fraud flag |

Derived features include aggregates, temporal signals, and WoE-transformed variables.

---

## Project Structure

```text
credit-risk-model/
├── config/                     # Centralized configuration
│   ├── data.yaml
│   ├── features.yaml
│   ├── model.yaml
│   ├── proxy.yaml
│   ├── train.yaml
│   └── api.yaml
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── external/
│
├── notebooks/
│   ├── eda.ipynb
│   ├── rfm.ipynb
│   ├── modeling.ipynb
│   └── experiments.ipynb
│
├── src/credit_risk_model/
│   ├── data/
│   │   ├── load_data.py
│   │   ├── preprocess.py
│   │   ├── rfm.py
│   │   ├── splitter.py
│   │   └── pipeline.py
│   │
│   ├── features/
│   │   ├── aggregate.py
│   │   ├── categorical.py
│   │   ├── numerical.py
│   │   ├── woe_iv.py
│   │   └── feature_builder.py
│   │
│   ├── models/
│   │   ├── train.py
│   │   ├── tuning.py
│   │   ├── evaluate.py
│   │   ├── predict.py
│   │   └── register.py
│   │
│   ├── api/
│   │   ├── main.py
│   │   ├── pydantic_models.py
│   │   └── utils.py
│   │
│   ├── pipeline/
│   │   ├── dvc_stage_data.py
│   │   ├── dvc_stage_features.py
│   │   ├── dvc_stage_train.py
│   │   └── dvc_stage_evaluate.py
│   │
│   └── utils/
│       ├── logger.py
│       ├── helpers.py
│       └── constants.py
│
├── tests/
│   ├── test_data_processing.py
│   ├── test_feature_engineering.py
│   ├── test_model_training.py
│   └── test_api.py
│
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── start.sh
│
├── scripts/
│   ├── run_api.sh
│   ├── run_training.sh
│   └── run_pipeline.sh
│
├── mlruns/
├── dvc.yaml
├── params.yaml
├── requirements.txt
└── README.md
```

## Modeling Approach

- Feature scaling and encoding implemented using `sklearn.pipeline` to ensure consistency between training and inference.
- Weight of Evidence (WoE) and Information Value (IV) used for feature selection and interpretability.
- Multiple models trained and compared, including linear and tree-based approaches.
- Model evaluation performed using:
  - **Accuracy**
  - **Precision / Recall**
  - **F1 Score**
  - **ROC-AUC**
- Hyperparameter optimization carried out using:
  - Grid Search
  - Random Search
- Best-performing model registered and versioned in **MLflow Model Registry**.

---

## MLOps & Engineering Practices

- **MLflow** for experiment tracking, metric logging, and model registry.
- **DVC** for data and pipeline reproducibility.
- **Pytest** for unit testing critical data and modeling components.
- Structured logging for traceability and debugging.
- CI-ready, modular project layout aligned with production standards.
- Fully **Dockerized API** for consistent deployment across environments.

---

## Setup & Installation

Clone the repository:

```bash
git clone https://github.com/<tib-dev>/credit-risk-model.git
cd credit-risk-model
```
### Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```
Install dependencies:

```bash

pip install -r requirements.txt
```
## Running the Project

### Run the full pipeline (DVC)
```bash
scripts/run_pipeline.sh
```

### Train models only
```bash
scripts/run_training.sh
```

### Start the API service
```bash
scripts/run_api.sh
```

### API Usage
#### Endpoint

##### POST /predict

#### Input

- Customer feature payload matching the trained model schema.

#### Output

    - Risk probability

    - Derived credit score

## Technologies Used

- Python
- scikit-learn
- MLflow
- DVC
- FastAPI
- Docker
- Pandas / NumPy
- Pytest

## Author

**Tibebu K.** – Analytics Engineer specializing Machine Learning, and Full-Stack Development  

- GitHub: [https://github.com/<tib-dev>](https://github.com/<tib-dev>)  
- LinkedIn: [https://www.linkedin.com/in/<tibeb>](https://www.linkedin.com/in/<tibeb>)  
- Email: <wise.tibec@gmail.com>
