import argparse, os
from utils.file_operations import read_lines
from shutil import copyfile, rmtree
from predict_tener_en import make_predictions
from java_wrapper import extract_dependencies_via_stanford
structure_stanford_results = __import__('structure-stanford-results')

parser = argparse.ArgumentParser()

parser.add_argument('--input', type=str, default='raw/text.txt')
parser.add_argument('--output', type=str, default='raw/text-entities.txt')

TMP_DATASET_DIR = 'tmp/conll2003ru-predicted'

args = parser.parse_args()

print("Extracting dependencies...")
# Extract dependencies
extract_dependencies_via_stanford(
	classpath = '/home/dima/CoreNLP/target/classes',
	embeddings = '/home/dima/models/ArModel100.txt',
	dependencies_model = '/home/dima/models/nndep.rus.modelAr100HS400.txt.gz',
	pos_model = '/home/dima/models/russian-ud-pos.tagger',
	input = args.input
)

print("Formatting dependencies...")
# Format dependencies
if not os.path.isdir(TMP_DATASET_DIR):
	os.mkdir(TMP_DATASET_DIR)
structure_stanford_results.structure_stanford_output(f'{args.input.split("/")[-1]}.out', f'{TMP_DATASET_DIR}/train.txt')
copyfile(f'{TMP_DATASET_DIR}/train.txt', f'{TMP_DATASET_DIR}/test.txt')
copyfile(f'{TMP_DATASET_DIR}/train.txt', f'{TMP_DATASET_DIR}/dev.txt')

print("Making predictions...")
# Make predictions
make_predictions(
	dataset_for_loading = 'conll2003ru-super-distinct',
	dataset_for_prediction = TMP_DATASET_DIR,
	filename = 'two-third',
	folderpath = '/home/dima/models/ner',
	subset_name_for_prediction = 'dev',
	output = args.output
)

rmtree(TMP_DATASET_DIR)
os.remove(f'{args.input.split("/")[-1]}.out')