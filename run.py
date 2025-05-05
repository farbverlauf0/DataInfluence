import os
import argparse
from utils import prepare_data, train_model_and_calculate_metrics, plot, SAMPLERS


ROOT_PATH = os.path.dirname(__file__)
DATA_PATH = os.path.join(ROOT_PATH, 'data')
MODELS_PATH = os.path.join(ROOT_PATH, 'models')
METRICS_PATH = os.path.join(ROOT_PATH, 'metrics')
GRAPHICS_PATH = os.path.join(ROOT_PATH, 'graphics')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-type')
    parser.add_argument('--sampler-type')
    parser.add_argument('--use-raw-data', action='store_true')
    parser.add_argument('--noise-scaler', type=float, default=0.0)
    args = parser.parse_args()
    if args.use_raw_data:
        prepare_data(
            root_path_to_data=DATA_PATH,
            data_type=args.data_type,
            noise_scaler=args.noise_scaler
        )
    else:
        print('Specify --use-raw-data for the correct results')
    if args.sampler_type == 'all':
        all_losses = {'train': {}, 'test': {}}
        for sampler_type in SAMPLERS:
            print(f'\n\nStarting sampler: {sampler_type}')
            losses = train_model_and_calculate_metrics(
                root_path_to_data=DATA_PATH,
                data_type=args.data_type,
                sampler_type=sampler_type,
                root_path_to_models=MODELS_PATH,
                root_path_to_metrics=METRICS_PATH
            )
            all_losses['train'][sampler_type] = losses['train']
            all_losses['test'][sampler_type] = losses['test']
        plot(
            all_losses=all_losses,
            base_path_to_graphics=GRAPHICS_PATH
        )
    else:
        train_model_and_calculate_metrics(
            root_path_to_data=DATA_PATH,
            data_type=args.data_type,
            sampler_type=args.sampler_type,
            root_path_to_models=MODELS_PATH,
            root_path_to_metrics=METRICS_PATH
        )
