import os
import pandas as pd
from sklearn.model_selection import train_test_split


SEED = 0
TEST_SIZE = 0.1


def prepare_zillow_prize(root_path_to_data: str):
    path_to_data = os.path.join(root_path_to_data, 'zillow_prize')
    path_to_raw_data = os.path.join(path_to_data, 'raw_data')
    objects = pd.read_csv(os.path.join(path_to_raw_data, 'properties_2016.csv'))
    targets = pd.read_csv(os.path.join(path_to_raw_data, 'train_2016_v2.csv'))
    targets = targets.iloc[:, :-1]
    targets = targets.groupby(by=['parcelid']).mean().reset_index(level='parcelid')
    data = pd.merge(targets, objects).iloc[:, 1:]
    mask = data.isna().mean() * 100 < 2
    cols = [feature for feature, add in zip(mask.index, mask) if add]
    data = data.loc[:, cols]
    for col in cols:
        if data[col].dtype != 'object':
            mean = float(data[col].mean())
        else:
            mean = data[col].mode()
        data[col].fillna(mean, inplace=True)
    data = data.dropna().rename(columns={'logerror': 'target'})
    train_data, test_data = train_test_split(data, test_size=TEST_SIZE, random_state=SEED)
    path_to_prepared_data = os.path.join(path_to_data, 'prepared_data')
    split_folder = os.path.join(path_to_prepared_data, 'split')
    if not os.path.exists(split_folder):
        os.mkdir(split_folder)
    train_data.to_csv(os.path.join(split_folder, 'train_data.csv'), index=False)
    test_data.to_csv(os.path.join(split_folder, 'test_data.csv'), index=False)
