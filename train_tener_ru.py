import argparse

from config_reader import get_config, get_valid_dataset_names
from utils.data_loaders import get_data_loading_function
from wrappers.model_factory import make_model

def train(config, models_path, training_dataset):
    make_model(model_id=config['model_ids']['tener'], dataset=training_dataset, config=config).train_model(models_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--training_dataset', type=str, default='en-ontonotes', choices=get_valid_dataset_names())
    parser.add_argument('--models_folder', type=str, default='/home/dima/models/ner')

    args = parser.parse_args()
    config = get_config(args.training_dataset)

    train(config=config, models_path=args.models_folder, training_dataset=args.training_dataset)