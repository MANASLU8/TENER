1. Raw text to text with dependencies
java -cp "/home/dima/CoreNLP/target/classes" edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,depparse -keepPunct edu.stanford.nlp.trees.international.russian.RussianTreebankLanguagePack \
-embedFile /home/dima/models/ArModel100.txt \
-embeddingSize 100 \
-parse EnhancedDependenciesAnnotation \
-depparse.model /home/dima/models/nndep.rus.modelAr100HS400.txt.gz \
-language Russian \
-textFile /home/dima/tener/src/raw/text.txt \
-outFile /home/dima/tener/src/tmp/dep-for-ner.txt \
-pos.model /home/dima/models/russian-ud-pos.tagger
2. Structure stanford results
python structure-stanford-results.py --input /home/dima/text.txt.out --output /home/dima/tener/src/tmp/output.txt