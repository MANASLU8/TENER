from utils.file_operations import read_lines, write_lines
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--input', type=str, default='/home/dima/text.txt.out')
#parser.add_argument('--pos', type=str, default='/home/dima/tener/src/tmp/pos.txt')
parser.add_argument('--output', type=str, default='/home/dima/tener/src/tmp/output.txt')

args = parser.parse_args()

def decode_dependency(dependency, pos_tags):
	#print(dependency)
	pair = dependency.split(', ')
	dependency_type, source = pair[0].split('(', 1)
	dest = pair[1].replace(')', '')
	splitted_source = ('-'.join(source.split('-')[:-1]), source.split('-')[-1])
	source_token_id = int(splitted_source[1])
	splitted_dest = ('-'.join(dest.split('-')[:-1]), dest.split('-')[-1])
	dest_token_id = int(splitted_dest[1])
	#print(dest_token_id, source_token_id, len(pos_tags))
	return (dependency_type, (splitted_source[0], int(splitted_source[1]), pos_tags[source_token_id]), (splitted_dest[0], int(splitted_dest[1]), pos_tags[dest_token_id]))

def make_dep_tree(root, tree, identation_level = 0, dep_tree = []):
	# print root word
	#print(tree)
	child_nodes = [dependency for dependency in tree if dependency[1][1] == root[1]]
	if (len(child_nodes) == 0):
		dep_tree.append(f'{" "*identation_level}("{root[0]}"/"{root[2]}"/{root[1]})')
	else:
		dep_tree.append(f'{" "*identation_level}("{root[0]}"/"{root[2]}"/{root[1]}')
		for dependency in child_nodes:
			#print(f'{" "*identation_level}{dependency[2]}')
			make_dep_tree(dependency[2], tree, identation_level = identation_level + 2, dep_tree = dep_tree)
		dep_tree.append(f'{" "*identation_level})')

phrase_types_mapping = {'VERB': 'VP', 'NOUN': 'NP'}

def get_phrase_type(node, tree):
	if node[1] == 0:
		return 'ROOT'
	elif node[2] in phrase_types_mapping:
		return phrase_types_mapping[node[2]]
	else:
		return get_phrase_type([another_node for another_node in tree if another_node[2][1] == node[1]][0][1], tree)

def get_bio_prefix(previous_phrase_type, current_phrase_type):
	if (previous_phrase_type == current_phrase_type):
		return 'I-'
	else:
		return 'B-'

def get_phrase_types(tree):
	previous_phrase_type = ''
	for i in range(len(tree)):
		for node in tree:
			if node[2][1] - 1 == i:
				current_phrase_type = get_phrase_type(node[2], tree)
				line_to_output = f'{node[2][0]} {get_bio_prefix(previous_phrase_type, current_phrase_type)}{current_phrase_type} {node[2][2]} O'
				previous_phrase_type = current_phrase_type
				yield line_to_output


def read_pos(pos_tag_file):
	sentences = []
	sentence = [None]
	previous_line = ''
	for line in read_lines(pos_tag_file)[:-1]:
		if not ((line == '!\tPUNCT') or (line == '.\tPUNCT') or (line == '?\tPUNCT')):
			sentence.append(line.split('\t')[1])
		else:
			if (len(sentence) != 1):
				sentence.append(line.split('\t')[1])
				sentences.append(sentence)
			sentence = [None]
		previous_line = line
	return sentences


STANFORD_DEPS_FILE = args.input
POS_FILE = args.output
PHRASE_TYPES_FILE = args.output

def structure_stanford_output(input_file, output_file):
	sentences = []
	sentence = []
	for line in read_lines(input_file):
		if not line.startswith('Sentence'):
			sentence.append(line)
			#sentence.append(decode_dependency(line))
		else:
			sentence = sentence[3:-1]
			#print(sentence)
			if len(sentence) == 0:
				continue
			#print(sentence[sentence.index('') + 2:])
			#print(sentence[3:sentence.index('')])
			#print(sentence[:sentence.index('')])
			# sentences.append({
			# 	'tokens': ['ROOT'] + list(map(lambda token: token.split('PartOfSpeech=')[1].replace(']', ''), sentence[:sentence.index('')])),
			# 	'dependencies': list(map(lambda i: decode_dependency(i, ['ROOT'] + list(map(lambda token: token.split('PartOfSpeech=')[1].replace(']', ''), sentence[:sentence.index('')]))), sentence[sentence.index('') + 2:]))
			# })
			sentences.append(list(map(lambda i: decode_dependency(i, ['ROOT'] + list(map(lambda token: token.split('PartOfSpeech=')[1].replace(']', ''), sentence[:sentence.index('')]))), sentence[sentence.index('') + 2:])))
			sentence = []
	sentence = sentence[3:-1]
	sentences.append(list(map(lambda i: decode_dependency(i, ['ROOT'] + list(map(lambda token: token.split('PartOfSpeech=')[1].replace(']', ''), sentence[:sentence.index('')]))), sentence[sentence.index('') + 2:])))
	#print(sentences[-1])
	#print(len(sentences))
	#print(sentences[0])

	dep_tree = []
	phrase_types = ['-DOCSTART- -X- -X- O', '']
	for sentence in sentences:
		root = ('ROOT', 0, 'ROOT')
		#print(f'({root[0]}//{root[1]}')
		make_dep_tree(root, sentence, 0, dep_tree)
		dep_tree.append('')
		for phrase_type in get_phrase_types(sentence):
			phrase_types.append(phrase_type)
		# Add empty line to split sentences
		phrase_types.append('')
		#print(dep_tree)
	#print(dep_tree)
	#print(list())
	#print(len(phrase_types))
	#write_lines("dependency_trees.txt", dep_tree)
	#read_lines(POS_FILE)


	write_lines(output_file, phrase_types)


	#pos_tags = read_pos(POS_FILE)
	#print(len(pos_tags))
	#print([i for i in pos_tags if len(i) == 2])
	#print(')')

if __name__ == "__main__":
	structure_stanford_output(STANFORD_DEPS_FILE, PHRASE_TYPES_FILE)