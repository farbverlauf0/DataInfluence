import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


SEED = 0
np.random.seed(SEED)
TEST_SIZE = 0.1


def prepare_zillow_prize(root_path_to_data: str, noise_scaler: float = 0.0):
    path_to_data = os.path.join(root_path_to_data, 'zillow_prize')
    path_to_raw_data = os.path.join(path_to_data, 'raw_data')
    objects = pd.read_csv(os.path.join(path_to_raw_data, 'properties_2016.csv'), low_memory=False)
    targets = pd.read_csv(os.path.join(path_to_raw_data, 'train_2016_v2.csv'), low_memory=False)
    targets = targets.iloc[:, :-1]
    targets = targets.groupby(by=['parcelid']).mean().reset_index(level='parcelid')
    data = pd.merge(targets, objects).iloc[:, 1:]
    data = data.loc[:, (data.isna().mean() < 0.1).to_numpy()]
    for col in data.columns:
        if data[col].dtype != 'object':
            mean = float(data[col].mean())
        else:
            mean = data[col].mode()
        data[col] = data[col].fillna(mean)
    data = data.dropna().rename(columns={'logerror': 'target'})
    drop_cols = [
        'calculatedbathnbr', 'fullbathcnt', 'latitude', 'longitude', 'propertycountylandusecode',
        'censustractandblock', 'regionidneighborhood', 'unitcnt', 'taxvaluedollarcnt', 'structuretaxvaluedollarcnt',
        'landtaxvaluedollarcnt', 'assessmentyear', 'airconditioningtypeid', 'heatingorsystemtypeid',
        'buildingqualitytypeid', 'propertyzoningdesc'
    ]
    data = data.drop(columns=drop_cols, errors='ignore')
    train_data, test_data = train_test_split(data, test_size=TEST_SIZE, random_state=SEED)

    noise_std = train_data.std().to_numpy().reshape(1, -1)
    noise = np.random.rand(*train_data.shape) * (noise_std * noise_scaler)
    train_data = train_data + noise

    path_to_prepared_data = os.path.join(path_to_data, 'prepared_data')
    split_folder = os.path.join(path_to_prepared_data, 'split')
    if not os.path.exists(split_folder):
        os.mkdir(split_folder)
    train_data.to_csv(os.path.join(split_folder, 'train_data.csv'), index=False)
    test_data.to_csv(os.path.join(split_folder, 'test_data.csv'), index=False)
