import os
import argparse
from utils import prepare_data, train_model, calculate_metrics


ROOT_PATH = os.path.dirname(__file__)
DATA_PATH = os.path.join(ROOT_PATH, 'data')
MODELS_PATH = os.path.join(ROOT_PATH, 'models')
METRICS_PATH = os.path.join(ROOT_PATH, 'metrics')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-type')
    parser.add_argument('--sampler-type')
    args = parser.parse_args()
    prepare_data(
        root_path_to_data=DATA_PATH,
        data_type=args.data_type
    )
    train_model(
        root_path_to_data=DATA_PATH,
        data_type=args.data_type,
        sampler_type=args.sampler_type,
        root_path_to_models=MODELS_PATH
    )
    calculate_metrics(
        root_path_to_models=MODELS_PATH,
        data_type=args.data_type,
        sampler_type=args.sampler_type,
        root_path_to_metrics=METRICS_PATH
    )
