import pandas as pd

import mlflow
import xgboost as xgb

import sklearn
from sklearn.model_selection import train_test_split

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

def read_data(file_path: str) -> pd.DataFrame:
    # Data Ingestion
    df = pd.read_csv(file_path)

    return df

def encode_data(df: pd.DataFrame) -> tuple:
    # Data Preprocessing/EDA

    # Defining categorical and numerical columns
    numerical_df = df[['id', 'age', 'bmi', 'sleep_hours']]
    categorical_cols = []
    for col in df.columns:
        if col not in numerical_df:
            categorical_cols.append(col)
        else:
            continue

    # Encoding categorical columns
    for column in categorical_cols:
        df[column] = df[column].astype('category')
        df[column] = df[column].cat.codes
    target = 'risk_level'
    Y_encoded = df[target]
    X_encoded = df.drop(columns=['id', target])

    return X_encoded, Y_encoded

class XGBoostModel:
    def __init__(self, X_train: pd.DataFrame, X_val: pd.DataFrame, Y_train: pd.DataFrame, Y_val: pd.DataFrame):
        self.X_train = X_train
        self.X_val = X_val
        self.Y_train = Y_train
        self.Y_val = Y_val

        self.search_space = {
        'objective': 'multi:softprob',
        'max_depth': scope.int(hp.quniform('max_depth', 4, 15, 1)),
        'learning_rate': hp.loguniform('learning_rate', -5, -1),
        'reg_alpha': hp.loguniform('reg_alpha', -5, 2),
        'reg_lambda': hp.loguniform('reg_lambda', -5, 2),
        'gamma': hp.loguniform('gamma', -3, 1),
        'min_child_weight': hp.loguniform('min_child_weight', 0, 3.5),
        'subsample': hp.uniform('subsample', 0.6, 1.0),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
        'seed': 42
    }
        self.default_params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'learning_rate': 0.1,
            'max_depth': 6,
            'min_child_weight': 1,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'seed': 42,
            'n_estimators': 1000,
            'early_stopping_rounds': 20,
            'eval_metric': 'mlogloss'
            }
    
    def objective(self, params: dict) -> dict:
        mlflow.xgboost.autolog()

        # Define the model (objective function) to minimize using hyperparameter tuning
        with mlflow.start_run():
            model = xgb.XGBClassifier(**params)
            model.fit(
                self.X_train,
                self.Y_train,
                eval_set=[(self.X_val, self.Y_val)]
                )
            Y_pred_proba = model.predict_proba(self.X_val)
            logloss = sklearn.metrics.log_loss(self.Y_val, Y_pred_proba)
            
        return {'loss': logloss, 'status': STATUS_OK}
    
    def param_tune(self, max_evals : int = 50) -> dict:
        opt_params = fmin(
            fn=self.objective,
            space=self.search_space, 
            algo=tpe.suggest,
            max_evals=max_evals, 
            trials=Trials())

        return opt_params
    
    def train_model(self, params: dict) -> xgb.XGBClassifier:
        # Build Model
        with mlflow.start_run() as run:
            mlflow.xgboost.autolog()
            
            if 'max_depth' in params and not isinstance(params['max_depth'], int):
                params['max_depth'] = int(params['max_depth'])
            
            model = xgb.XGBClassifier(
                objective='multi:softprob',
                num_class=3,
                n_estimators=1000,
                eval_metric='mlogloss',
                early_stopping_rounds=20,
                seed=42,
                **params      
            )

            model.fit(
                self.X_train,
                self.Y_train,
                eval_set=[(self.X_val, self.Y_val), (self.X_train, self.Y_train)]
            )

            Y_pred_proba = model.predict_proba(self.X_val)
            logloss = sklearn.metrics.log_loss(self.Y_val, Y_pred_proba)
            print('The predicted values are: ', Y_pred_proba, 'and the losses are: ', logloss)

            return run.info.run_id, model

def run(file_path: str) -> xgb.XGBClassifier:
    df = read_data(file_path)
    X_encoded, Y_encoded = encode_data(df)
    X_train, X_val, Y_train, Y_val = train_test_split(X_encoded, Y_encoded)
    model = XGBoostModel(X_train, X_val, Y_train, Y_val)
    opt_param = model.param_tune()
    run_id, opt_model = model.train_model(opt_param)

    return run_id, opt_model