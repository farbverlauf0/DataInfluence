from .zillow_prize import prepare_zillow_prize


MAP = {
    'zillow_prize': prepare_zillow_prize
}


def prepare_data(root_path_to_data: str, data_type: str, noise_scaler: float):
    if data_type not in MAP:
        raise ValueError(f'Incorrect data_type: {data_type}\nAvailable types: {", ".join(list(MAP.keys()))}')
    return MAP[data_type](root_path_to_data, noise_scaler)
