#!/bin/sh
NUMBER_OF_FOLDS=${1:-10}

NER_COMPARISON_HOME=/home/nami/ner-comparison

cd /home/nami/TENER
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
deactivate
. ./python/bin/activate

mkdir $NER_COMPARISON_HOME/cv/tener
for i in $(seq -f "%02g" 1 $NUMBER_OF_FOLDS)
do
	python predict_tener_ru.py --training_dataset data/conll2003rucv-$i --prediction_dataset data/conll2003rucv-$i --model_file /home/nami/models/ner/$i --subset test --output_file $NER_COMPARISON_HOME/cv/tener/$i.txt
done

#python predict_tener_ru.py --training_dataset conll2003ru-bio-super-distinct --prediction_dataset conll2003ru-bio-super-distinct --model_file /home/dima/models/ner/bio  --subset dev --output_file preds.txt