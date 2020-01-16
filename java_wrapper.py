import subprocess

def extract_dependencies_via_stanford(classpath, embeddings, dependencies_model, pos_model, input):
	command = [
		'java',
		'-cp', classpath, 'edu.stanford.nlp.pipeline.StanfordCoreNLP',
		'-annotators', 'tokenize,ssplit,pos,depparse',
		'-keepPunct',
		'edu.stanford.nlp.trees.international.russian.RussianTreebankLanguagePack'
		'-embedFile', embeddings,
		'-embeddingSize', '100',
		'-parse', 'EnhancedDependenciesAnnotation',
		'-depparse.model', dependencies_model,
		'-language', 'Russian',
		'-textFile', input,
		'-outFile', '/home/dima/tener/src/tmp/dep-for-ner.txt',
		'-pos.model', pos_model
	]
	#print(' '.join(command))
	stdout, stderr = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).communicate()
	print(f'stdout: {stdout}')
	print(f'stderr: {stderr}')