import argparse

from config_reader import get_config, get_valid_dataset_names
from utils.data_loaders import get_data_loading_function
from wrappers.model_factory import make_model

def predict(config, model_path, prediction_dataset, prediction_subset, output_path):
  make_model(model_id=config['model_ids']['tener'], dataset=prediction_dataset, config=config).load(model_path).export_predictions(prediction_dataset, prediction_subset, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    #parser.add_argument('--training_dataset', type=str, default='en-ontonotes')
    parser.add_argument('--prediction_dataset', type=str, default='tmp/conll2003ru-predicted')
    parser.add_argument('--model_file', type=str, default='/home/dima/models/ner/bio')
    parser.add_argument('--output_file', type=str, default='predictions.txt')
    parser.add_argument('--subset', type=str, default='test')

    args = parser.parse_args()
    config = get_config(args.prediction_dataset)

    predict(config=config, model_path=args.model_file, prediction_dataset=args.prediction_dataset, prediction_subset=args.subset, output_path=args.output_file)
