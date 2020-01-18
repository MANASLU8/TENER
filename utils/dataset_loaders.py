from modules.pipe import Conll2003NERPipe

def read_dataset(dataset, config):
  if dataset == 'conll2003':
      return read_conll_dataset('../data/conll2003', config)
  if dataset == 'conll2003ru':
      return read_conll_dataset('../data/conll2003ru', config)
  elif dataset == 'conll2003ru-distinct':
      return read_conll_dataset('../data/conll2003ru-distinct', config)
  elif dataset == 'conll2003ru-super-distinct':
      return read_conll_dataset('../data/conll2003ru-super-distinct', config)
  elif dataset == 'conll2003ru-bio-super-distinct':
      return read_conll_dataset('../data/conll2003ru-bio-super-distinct', config)
  elif dataset == 'conll2003ru-big':
      return read_conll_dataset('../data/conll2003ru-big', config)
  elif dataset == 'en-ontonotes':
      # 会使用这个文件夹下的train.txt, test.txt, dev.txt等文件
      paths = '../data/en-ontonotes/english'
      return OntoNotesNERPipe(encoding_type=encoding_type).process_from_file(paths)
  else:
  	  return read_conll_dataset(dataset, config)

def read_conll_dataset(root_dir, config):
    # conll2003的lr不能超过0.002
    paths = {
                'test': f"{root_dir}/test.txt",
                'train': f"{root_dir}/train.txt",
                'dev': f"{root_dir}/dev.txt"
             }
    data = Conll2003NERPipe(encoding_type=config['encoding_type'])
    data = data.process_from_file(paths)
    data.rename_field('words', 'chars')
    return data