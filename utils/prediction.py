from fastNLP.core.predictor import Predictor
from utils.file_operations import write_lines

def predict(model, subset_for_prediction, targets, filename):
	predictor = Predictor(model)
	predictions = predictor.predict(subset_for_prediction)['pred']
	words = list(subset_for_prediction.get_field('raw_words'))
	lines = []
	# print(predictions)
	# print(f'predicted labels for {len(predictions)}/{len(words)} items')

	words_sequence_index = 1
	labels_sequence_index = 0
	for sentence in list(zip(predictions, words)):
	  words = sentence[words_sequence_index]
	  labels = map(lambda label: f'{targets.to_word(label).split("-")[-1]}', sentence[labels_sequence_index][0])
	  for pair in zip(words, labels):
	    lines.append('\t'.join(pair))
	  lines.append('')

	write_lines(filename, lines)
