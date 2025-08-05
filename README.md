# Prostate Cancer Risk Prediction - MLOps Pipeline

[![CI/CD Pipeline](https://github.com/KevinKBui/Prostate-Cancer-Risk-Assessment/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/KevinKBui/Prostate-Cancer-Risk-Assessment/actions/workflows/ci-cd.yml)

A complete end-to-end MLOps pipeline for predicting prostate cancer risk levels based on patient health indicators and lifestyle factors. This project demonstrates modern MLOps practices including experiment tracking, workflow orchestration, model deployment, monitoring, and CI/CD.

## Project Overview

Prostate cancer is a rising issue within society. There is an alarming increase in the rate of occurrence of prostate cancer and the medical industry is not currently equipped to handle the population size at hand. Additionally, because cancer is often hidden within the body, it becomes increasingly difficult to diagnose and treat. We do not currently have safe, affordable and effective diagnostic methods that can detect and determine which individuals will be affected by prostate cancer. To augment the current healthcare system and its providers, this project was developed to evaluate which individuals are most at-risk of experiencing prostate cancer. To achieve this task, this project implements a machine learning system to predict prostate cancer risk levels (Low, Medium, High) using patient data. The patient data includes:

- **Demographics**: Age
- **Health Metrics**: BMI, Sleep Hours
- **Medical History**: Family History
- **Lifestyle Factors**: Diet, Exercise Frequency, Smoking Status, Alcohol Consumption

### MLOps Components Implemented

**Experiment Tracking & Model Registry** (MLflow)  
**Workflow Orchestration** (Prefect)  
**Model Deployment** (Flask + Docker)  
**Model Monitoring** (Evidently AI)  
**Best Practices** (Testing, Linting, Pre-commit hooks)  
**CI/CD Pipeline** (GitHub Actions)  
**Infrastructure as Code** (Docker Compose)  

## How to Get Started

The instructions in this README are intended for Linux systems.  

### 1. Setup Environment

Clone the repository
```bash
git clone https://github.com/KevinKBui/Prostate-Cancer-Risk-Assessment.git
```

Change the directory to the Repository folder
```bash
cd Prostate-Cancer-Risk-Assessment
```

Activate virtual environment
```bash
source mlops_env/bin/activate
```

Setup MLOps environment (install dependencies and create directories)
```bash
make setup-mlops
```

### 2. Train the Model

Train model using Prefect orchestration (uses local MLflow tracking)
```bash
make train
```

### 3. Start Web Service

Start the prediction web service
```bash
make serve
```

### 4. Test the Service

The web service will be available at:
- **Web Interface**: http://localhost:9696
- **Health Check**: http://localhost:9696/health
- **API Endpoint**: http://localhost:9696/predict_api

Test with curl:
```bash
curl -X POST http://localhost:9696/predict_api \
  -H "Content-Type: application/json" \
  -d '{
    "age": 45,
    "bmi": 25.5,
    "sleep_hours": 7.5,
    "family_history": "Yes",
    "diet": "Healthy",
    "exercise_frequency": "Daily",
    "smoking_status": "Never",
    "alcohol_consumption": "Light"
  }'
```

### 5. Optional: Docker Deployment

If you have Docker installed:
```bash
# Build and deploy all services
make prod-deploy
```

### 6. Access Services

- **Web Service**: http://localhost:9696
- **MLflow UI**: http://localhost:5000 (if using Docker deployment)
- **Grafana**: http://localhost:3000 (if using Docker deployment)
- **Database Admin**: http://localhost:8080 (if using Docker deployment)

## Model Training & Experiment Tracking

### Training Pipeline

The training pipeline uses **Prefect** for orchestration and **MLflow** for experiment tracking:

Run training pipeline
```bash
python src/orchestration.py
```

**Pipeline Steps:**
1. **Data Loading**: Load dataset from CSV
2. **Preprocessing**: Encode categorical variables, split features/target
3. **Data Splitting**: Train/validation split with stratification
4. **Hyperparameter Optimization**: Hyperopt with XGBoost
5. **Model Training**: Train final model with best parameters
6. **Model Registration**: Save to MLflow registry

### Experiment Tracking

All experiments are tracked in MLflow with:
- **Parameters**: Hyperparameters, data splits, model parameters, model performance metrics (logloss error)
- **Metrics**: Accuracy, precision, recall, F1-score
- **Artifacts**: Trained models, preprocessing objects
- **Model Registry**: Versioned model storage

View experiments at: http://localhost:5000

## Model Deployment

### Web Service

The model is deployed as a **Flask web service** with both web interface and API endpoints:

Start web service locally
```bash
make serve
```

**Endpoints:**
- `GET /`: Web form for predictions
- `POST /predict_api`: JSON API for predictions
- `GET /health`: Health check
- `GET /model_info`: Model metadata

**Example API Usage:**
```bash
curl -X POST http://localhost:9696/predict_api \
  -H "Content-Type: application/json" \
  -d '{
    "age": 45,
    "bmi": 25.5,
    "sleep_hours": 7.5,
    "family_history": "Yes",
    "diet": "Healthy",
    "exercise_frequency": "Daily",
    "smoking_status": "Never",
    "alcohol_consumption": "Light"
  }'
```

### Docker Deployment

Build and Deploy Docker image
```bash
make build
make deploy
```

## Model Monitoring

### Monitoring System

The monitoring system uses **Evidently AI** to track:
- **Data Drift**: Distribution changes in input features
- **Model Performance**: Accuracy degradation over time
- **Data Quality**: Missing values, data types, duplicates
- **Prediction Logs**: All predictions with timestamps

### Running Monitoring

```bash
# Run monitoring analysis
make monitor

# Generate test predictions for monitoring
make generate-test-data
```

### Monitoring Dashboard

Access the monitoring dashboard at: `monitoring/dashboard.html`

**Features:**
- Data drift detection and alerts
- Model performance metrics
- Prediction volume tracking
- Historical trend analysis

## Testing (Unit and Integration Tests) and Formatting (Linting)

### Running Tests

```bash
# Unit tests
make test

# Integration tests
make integration-tests

# Code quality checks
make quality-checks
```

### Test Coverage

- **Unit Tests**: Model functions, preprocessing, API endpoints
- **Integration Tests**: End-to-end pipeline testing
- **Performance Tests**: Large dataset processing
- **Edge Case Tests**: Invalid inputs, boundary conditions

### Code Quality

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **pre-commit**: Automated quality checks

## CI/CD Pipeline

### GitHub Actions Workflow

The CI/CD pipeline includes:

1. **Code Quality**: Linting, formatting
2. **Testing**: Unit tests, integration tests, model validation
3. **Docker**: Build and push container images
4. **Deployment**: Automated deployment to production

## Monitoring & Observability

### Metrics Tracked

- **Model Performance**: Accuracy, precision, recall, F1-score
- **Data Quality**: Missing values, data types, duplicates
- **Data Drift**: Feature distribution changes
- **Service Health**: Response times, error rates

### Alerting

- Data drift detection alerts
- Model performance degradation
- Service health issues
- Data quality problems

## Development Workflow

### Local Development

Setup development environment
```bash
make dev-setup
```

Run quality checks
```bash
make quality-checks
```

Run tests
```bash
make test
```

Commit changes (pre-commit hooks will run)
```bash
git add .
git commit -m "Your changes"
```
