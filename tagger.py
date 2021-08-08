import os
import argparse
import logging
from flair.models import SequenceTagger
from flair.datasets.sequence_labeling import ColumnDataset
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

logging.basicConfig(level=logging.DEBUG, format='%(process)d-%(levelname)s-%(message)s')
tested_models = ['bert-base', 'elmo-small', 'finbert-combo', 'finbert-prosusai', 'roberta-model']

def predict(model_path, input_file, output_file):
    logging.info(f'Testing model {model_path}')
    try:
        model = SequenceTagger.load(os.path.join(model_path, 'final-model.pt'))
    except Exception as e:
        print(f'Error while loading model from {model_path}: {str(e)}')
        return
    columns = {0 : 'text', 1 : 'ner'}
    dataset = ColumnDataset(input_file, columns)
    sentences = dataset.sentences
    tokens = lambda sent: pd.DataFrame(list(map(lambda _: (_.text, _.labels[0].value, _.labels[0].score), sent.tokens)),
                                       columns=['token', 'label', 'score'])
    actual = pd.concat(list(map(lambda _: tokens(_[1]).assign(sentence=_[0]), enumerate(sentences))))\
        .reset_index(drop=True).reset_index()
    actual = actual.drop(labels='score', axis=1)
    model.predict(sentences)
    predicted = pd.concat(list(map(lambda _: tokens(_[1]).assign(sentence=_[0]), enumerate(sentences))))\
        .reset_index(drop=True).reset_index()
    results: pd.DataFrame = actual.merge(predicted, on='index')[['sentence_x', 'token_x', 'label_x', 'label_y', 'score']]
    results.columns = ['sentence', 'token', 'actual', 'predicted', 'score']
    logging.info(accuracy_score(results['actual'], results['predicted']))
    logging.info(classification_report(results['actual'], results['predicted']))
    results.to_csv(output_file, index=False, sep=';')


def tag(model_path, input_file, output_file):
    logging.info(f'Testing model {model_path}')
    columns = {0 : 'text'}
    dataset = ColumnDataset(input_file, columns)
    sentences = dataset.sentences
    tokens = lambda sent: pd.DataFrame(list(map(lambda _: (_.text, _.labels[0].value, _.labels[0].score), sent.tokens)),
                                       columns=['token', 'label', 'score'])
    try:
        model = SequenceTagger.load(os.path.join(model_path, 'final-model.pt'))
    except Exception as e:
        print(f'Error while loading model from {model_path}: {str(e)}')
        return
    model.predict(sentences)
    results = pd.concat(list(map(lambda _: tokens(_[1]).assign(sentence=_[0]), enumerate(sentences))))\
        .reset_index(drop=True).reset_index(drop=True)
    results.columns = ['token', 'predicted', 'score', 'sentence']
    results.to_csv(output_file, index=False, sep=';')


def create_predictions(input_file, prefix='test', perform_tagging=False):
    if not os.path.isdir('results'):
        os.mkdir('results')
    for model in tested_models:
        if perform_tagging is True:
            tag(model, input_file, os.path.join('results', f'{prefix}-{model}.csv'))
        else:
            predict(model, input_file, os.path.join('results', f'{prefix}-{model}.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", help="Test data input path")
    parser.add_argument("--tagging-mode", help="Perform tagging", dest='tagging_mode', action='store_true')
    parser.set_defaults(tagging_mode=False)
    args = parser.parse_args()
    fpath = os.path.join(args.input_path, 'test.txt')
    if os.path.isfile(fpath):
        prefix = 'tag' if args.tagging_mode is True else 'test'
        create_predictions(fpath, prefix, perform_tagging=args.tagging_mode)
    fpath = os.path.join(args.input_path, 'train.txt')
    if os.path.isfile(fpath):
        create_predictions(fpath, 'train', perform_tagging=args.tagging_mode)