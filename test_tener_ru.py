import argparse

from config_reader import get_config, get_valid_dataset_names
from utils.data_loaders import get_data_loading_function
from wrappers.model_factory import make_model

def test(config, model_path, training_dataset, testing_dataset, testing_subset):
    make_model(model_id=config['model_ids']['tener'], dataset=training_dataset, config=config).load(model_path).test(testing_dataset, testing_subset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--training_dataset', type=str, default='en-ontonotes', choices=get_valid_dataset_names())
    parser.add_argument('--testing_dataset', type=str, default='tmp/conll2003ru-predicted')
    parser.add_argument('--model_file', type=str, default='/home/dima/models/ner/bio')
    parser.add_argument('--subset', type=str, default='test')

    args = parser.parse_args()
    config = get_config(args.training_dataset)

    test(config=config, model_path=args.model_file, training_dataset=args.training_dataset, testing_dataset=args.testing_dataset, testing_subset=args.subset)