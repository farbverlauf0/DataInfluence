import os
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from .samplers import BaseSampler, RandomSampler


SEED = 0
SAMPLERS = {
    'base': BaseSampler,
    'random': RandomSampler
}


def train_model_and_calculate_metrics(root_path_to_data: str, data_type: str, sampler_type: str,
                                      root_path_to_models: str, root_path_to_metrics: str):
    path_to_data = os.path.join(root_path_to_data, data_type)
    path_to_prepared_data = os.path.join(path_to_data, 'prepared_data')
    train_data = pd.read_csv(os.path.join(path_to_prepared_data, 'split', 'train_data.csv'))
    x_train = train_data.iloc[:, 1:].to_numpy()
    y_train = train_data.iloc[:, 0].to_numpy()
    if sampler_type not in SAMPLERS.keys():
        raise ValueError('Select the right sampler')
    num_samples = int(len(x_train) * 0.1)
    sampler = SAMPLERS[sampler_type](num_samples=num_samples)
    x_train, y_train, _ = sampler(x_train, y_train, np.ones_like(x_train))

    def objective(trial):
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0, log=True),
            'depth': trial.suggest_int('depth', 2, 10),
            'iterations': 1000,
            'random_state': SEED,
            'logging_level': 'Silent'
        }
        regressor = CatBoostRegressor(**params)
        scores = cross_val_score(regressor, x_train, y_train, scoring='neg_mean_absolute_error', cv=9)
        return scores.mean()

    study = optuna.create_study(study_name=f'catboost-seed{SEED}', direction='maximize')
    study.optimize(objective, n_trials=30)
    model = CatBoostRegressor(**study.best_params)
    model.fit(x_train, y_train)
    path_to_model = os.path.join(root_path_to_models, f'model_{data_type}_{sampler_type}')
    model.save_model(path_to_model)
    test_data = pd.read_csv(os.path.join(path_to_prepared_data, 'split', 'test_data.csv'))
    x_test = test_data.iloc[:, 1:].to_numpy()
    y_test = test_data.iloc[:, 0].to_numpy()
    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    with open(os.path.join(root_path_to_metrics, f'metrics_{data_type}_{sampler_type}.txt'), 'w') as f:
        f.write(f'MSE: {mse}\nMAE: {mae}\nR2: {r2}')
