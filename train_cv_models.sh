#!/bin/sh
NUMBER_OF_FOLDS=${1:-10}

cd /home/nami/TENER
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
deactivate
. ./python/bin/activate

for i in $(seq -f "%02g" 1 $NUMBER_OF_FOLDS)
do
	python train_tener_ru.py --training_dataset data/conll2003rucv-$i --models_folder /home/nami/models/ner/tener-cv
done
