import argparse
from utils.file_operations import read_lines

parser = argparse.ArgumentParser()

parser.add_argument('--text', type=str, default='raw/text.txt')
# parser.add_argument('--filename', type=str, default='best_TENER_f_2020-01-15-17-40-12')
# parser.add_argument('--folderpath', type=str, default='/home/dima/models/ner')
# parser.add_argument('--subset', type=str, default='test')
# parser.add_argument('--output', type=str, default='predictions/test.txt')

args = parser.parse_args()

text = '\n'.join(read_lines(args.text))

print(text)