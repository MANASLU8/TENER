import yaml
from utils.file_operations import read_yaml

CONFIG_PATH = 'config.yaml'
DATASETS_PATH = 'datasets.yaml'

def _get_dataset_dependent_params(dataset):
    if dataset in datasets['conll']:
        return {
            'n_heads': 14,
            'head_dims': 128,
            'num_layers': 2,
            'lr': 0.0009,
            'attn_type': 'adatrans',
            'char_type': 'cnn'
        }
    elif dataset == 'en-ontonotes':
        return {
            'n_heads': 8,
            'head_dims': 96,
            'num_layers': 2,
            'lr': 0.0007,
            'attn_type': 'adatrans',
            'char_type': 'adatrans'
        }

def get_config(dataset):
    config = read_yaml(CONFIG_PATH)
    config.update(_get_dataset_dependent_params(dataset))
    config.update({'datasets': datasets})
    return config

def get_valid_dataset_names():
    return list(set([dataset for datasets_list_label in datasets for dataset in datasets[datasets_list_label]]))

def get_datasets():
    return read_yaml(DATASETS_PATH)

datasets = get_datasets()
