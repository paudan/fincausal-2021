#!/bin/bash

FEATURES_PATH=features/evaluation
mkdir $FEATURES_PATH
mkdir results
python3 dataset_extractor.py --input-path data/task2.csv --output-path $FEATURES_PATH/task2-dataset.csv --use-iobe
python3 splitter.py --input-path $FEATURES_PATH/task2-dataset.csv --output-path $FEATURES_PATH --test-size 1
python3 tagger.py --input-path $FEATURES_PATH --tagging-mode
python3 evaluator.py --test-file results/tag-elmo-small.csv --source-file data/task2.csv --output-file submissions/submission-elmo.zip
python3 evaluator.py --test-file results/tag-bert-base.csv --source-file data/task2.csv --output-file submissions/submission-bert.zip
python3 evaluator.py --test-file results/tag-finbert-prosusai.csv --source-file data/task2.csv --output-file submissions/submission-finbert-prosusai.zip
python3 stacker.py
python3 evaluator.py --test-file results/tag-stacked.csv --source-file data/task2.csv --output-file submissions/submission-stacked.zip
python3 evaluator.py --test-file results/tag-voting.csv --source-file data/task2.csv --output-file submissions/submission-voting.zip