import argparse, os
from utils.file_operations import read_lines
from shutil import copyfile, rmtree
from predict_tener_ru import predict
from config_reader import get_config
from java_wrapper import extract_dependencies_via_stanford
structure_stanford_results = __import__('structure-stanford-results')

TMP_DATASET_DIR = 'tmp/conll2003ru-predicted'
DATASET = 'conll2003ru-bio-super-distinct'

def extract(input_file, output_file):
    print("Extracting dependencies...")
    # Extract dependencies
    extract_dependencies_via_stanford(
        classpath = '/home/dima/CoreNLP/target/classes',
        embeddings = '/home/dima/models/ArModel100.txt',
        dependencies_model = '/home/dima/models/nndep.rus.modelAr100HS400.txt.gz',
        pos_model = '/home/dima/models/russian-ud-pos.tagger',
        input = input_file
    )

    print("Formatting dependencies...")
    # Format dependencies
    if not os.path.isdir(TMP_DATASET_DIR):
        os.mkdir(TMP_DATASET_DIR)
    structure_stanford_results.structure_stanford_output(f'{input_file.split("/")[-1]}.out', f'{TMP_DATASET_DIR}/train.txt')
    copyfile(f'{TMP_DATASET_DIR}/train.txt', f'{TMP_DATASET_DIR}/test.txt')
    copyfile(f'{TMP_DATASET_DIR}/train.txt', f'{TMP_DATASET_DIR}/dev.txt')

    print("Making predictions...")
    # Make predictions
    predict(
        config = get_config(DATASET),
        model_path = '/home/dima/models/ner/bio',
        training_dataset = DATASET,
        prediction_dataset = TMP_DATASET_DIR,
        prediction_subset = 'dev',
        output_path = output_file
    )
    
    rmtree(TMP_DATASET_DIR)
    os.remove(f'{input_file.split("/")[-1]}.out')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str, default='raw/text.txt')
    parser.add_argument('--output', type=str, default='raw/text-entities.txt')

    args = parser.parse_args()
    extract(args.input, args.output)
