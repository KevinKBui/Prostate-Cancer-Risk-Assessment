import logging
import pickle
from pathlib import Path

import mlflow
import pandas as pd
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from prefect import flow, task
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@task
def load_data(file_path: str) -> pd.DataFrame:
    """Load data from CSV file"""
    logger.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    logger.info(f"Data loaded with shape: {df.shape}")
    return df


@task
def preprocess_data(df: pd.DataFrame) -> tuple:
    """Preprocess the data for training"""
    logger.info("Starting data preprocessing")

    # Defining categorical and numerical columns
    numerical_cols = ["age", "bmi", "sleep_hours"]
    categorical_cols = [
        col for col in df.columns if col not in numerical_cols + ["id", "risk_level"]
    ]

    # Encoding categorical columns
    df_processed = df.copy()
    for column in categorical_cols:
        df_processed[column] = df_processed[column].astype("category")
        df_processed[column] = df_processed[column].cat.codes

    target = "risk_level"
    if target in df_processed.columns:
        Y_encoded = df_processed[target]
        X_encoded = df_processed.drop(
            columns=["id", target] if "id" in df_processed.columns else [target]
        )
    else:
        # For inference, no target column
        Y_encoded = None
        X_encoded = df_processed.drop(
            columns=["id"] if "id" in df_processed.columns else []
        )

    logger.info(f"Features shape: {X_encoded.shape}")
    if Y_encoded is not None:
        logger.info(f"Target shape: {Y_encoded.shape}")

    return X_encoded, Y_encoded


@task
def split_data(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42
) -> tuple:
    """Split data into train and validation sets"""
    logger.info("Splitting data into train and validation sets")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(
        f"Train set shape: {X_train.shape}, Validation set shape: {X_val.shape}"
    )
    return X_train, X_val, y_train, y_val


@task
def hyperparameter_optimization(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    max_evals: int = 50,
) -> dict:
    """Optimize hyperparameters using Hyperopt"""
    logger.info("Starting hyperparameter optimization")

    search_space = {
        "max_depth": scope.int(hp.quniform("max_depth", 4, 15, 1)),
        "learning_rate": hp.loguniform("learning_rate", -5, -1),
        "reg_alpha": hp.loguniform("reg_alpha", -5, 2),
        "reg_lambda": hp.loguniform("reg_lambda", -5, 2),
        "gamma": hp.loguniform("gamma", -3, 1),
        "min_child_weight": hp.loguniform("min_child_weight", 0, 3.5),
        "subsample": hp.uniform("subsample", 0.6, 1.0),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.6, 1.0),
    }

    def objective(params):
        with mlflow.start_run(nested=True):
            model = xgb.XGBClassifier(
                objective="multi:softprob",
                num_class=3,
                n_estimators=100,
                eval_metric="mlogloss",
                early_stopping_rounds=10,
                seed=42,
                **params,
            )

            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            y_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)

            mlflow.log_params(params)
            mlflow.log_metric("accuracy", accuracy)

            return {"loss": -accuracy, "status": STATUS_OK}

    trials = Trials()
    best_params = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
    )

    logger.info(f"Best parameters: {best_params}")
    return best_params


@task
def train_model(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    best_params: dict,
) -> tuple:
    """Train the final model with best parameters"""
    logger.info("Training final model")

    # Convert float to int for max_depth
    if "max_depth" in best_params:
        best_params["max_depth"] = int(best_params["max_depth"])

    with mlflow.start_run() as run:
        model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            n_estimators=1000,
            eval_metric="mlogloss",
            early_stopping_rounds=20,
            seed=42,
            **best_params,
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val), (X_train, y_train)],
            verbose=False,
        )

        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)

        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_pred_train)
        val_accuracy = accuracy_score(y_val, y_pred_val)

        # Log parameters and metrics
        mlflow.log_params(best_params)
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("val_accuracy", val_accuracy)

        # Log model
        mlflow.xgboost.log_model(model, "model")

        # Save model locally
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        model_path = models_dir / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        logger.info(
            f"Model trained. Train accuracy: {train_accuracy:.4f}, Val accuracy: {val_accuracy:.4f}"
        )
        logger.info(f"Model saved to {model_path}")

        return run.info.run_id, model


@task
def save_run_id(run_id: str):
    """Save the run ID for later use"""
    run_id_folder = Path("models/run_id")
    run_id_folder.mkdir(parents=True, exist_ok=True)
    with open(run_id_folder / "run_id.txt", "w") as f:
        f.write(run_id)
    logger.info(f"Run ID {run_id} saved")


@flow(name="training-pipeline")
def training_pipeline(
    file_path: str = "./Data/raw/synthetic_prostate_cancer_risk.csv",
):
    """Main training pipeline"""
    logger.info("Starting training pipeline")

    # Set MLflow tracking URI to local file-based tracking
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Prostate Cancer Risk Prediction")

    # Load and preprocess data
    df = load_data(file_path)
    X, y = preprocess_data(df)

    # Split data
    X_train, X_val, y_train, y_val = split_data(X, y)

    # Optimize hyperparameters
    best_params = hyperparameter_optimization(X_train, X_val, y_train, y_val)

    # Train final model
    run_id, model = train_model(X_train, X_val, y_train, y_val, best_params)

    # Save run ID
    save_run_id(run_id)

    logger.info("Training pipeline completed")
    return run_id


@flow(name="inference-pipeline")
def inference_pipeline(data_path: str, model_path: str = "models/model.pkl"):
    """Inference pipeline for making predictions"""
    logger.info("Starting inference pipeline")

    # Load data
    df = load_data(data_path)
    X, _ = preprocess_data(df)

    # Load model
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Make predictions
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    # Create results DataFrame
    results = df.copy()
    results["prediction"] = predictions
    results["probability_low"] = probabilities[:, 0]
    results["probability_medium"] = probabilities[:, 1]
    results["probability_high"] = probabilities[:, 2]

    # Save results
    output_path = "predictions.csv"
    results.to_csv(output_path, index=False)

    logger.info(f"Inference completed. Results saved to {output_path}")
    return output_path


if __name__ == "__main__":
    # Run training pipeline
    training_pipeline()
