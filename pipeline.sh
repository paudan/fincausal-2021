#!/bin/bash

FEATURES_PATH=features/combined
python3 dataset_extractor.py --input-path data/combined_fnp2020-fincausal-task2.csv \
    --output-path $FEATURES_PATH/combined_fnp2020-fincausal-task2-dataset.csv \
    --use-iobe
python3 splitter.py --input-path $FEATURES_PATH/combined_fnp2020-fincausal-task2-dataset.csv --output-path $FEATURES_PATH \
   --test-size 0.2
python3 trainer.py --input-path $FEATURES_PATH
python3 tagger.py --input-path $FEATURES_PATH
python3 classifier.py --train-file $FEATURES_PATH/train.csv --test-file $FEATURES_PATH/test.csv
python3 evaluator.py
python3 stacker.py