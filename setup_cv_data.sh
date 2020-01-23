#!/bin/sh
NUMBER_OF_FOLDS=${1:-10}

for i in $(seq -f "%02g" 1 $NUMBER_OF_FOLDS)
do
	cp -r /home/dima/fact-ru-eval2stanford/conll2003rucv/$i /home/dima/tener/data/conll2003rucv-$i
	cp /home/dima/tener/data/conll2003rucv-$i/test.txt /home/dima/tener/data/conll2003rucv-$i/dev.txt
done