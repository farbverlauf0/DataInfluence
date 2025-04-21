import os
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from .samplers import BaseSampler, RandomSampler


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
    sampler = SAMPLERS[sampler_type]()
    x_train, y_train, _ = sampler(x_train, y_train, ...)
    model = CatBoostRegressor()
    model.fit(x_train, y_train, cat_features=[8])
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
