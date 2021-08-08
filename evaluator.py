import os
import glob
import zipfile
import argparse
import numpy as np
import pandas as pd


def extract_spans(group: pd.DataFrame):

    def extract_spans(column):
        index_ = lambda vals, label: np.argwhere(vals.values == label).flatten()
        effect_spans = list(zip(index_(group[column], 'B-E'), index_(group[column], 'E-E')))
        causal_spans = list(zip(index_(group[column], 'B-C'), index_(group[column], 'E-C')))
        effects = [(list(range(start, end+1)), group['token'].iloc[start:end+1].tolist()) for start, end in effect_spans]
        if not effects:
            effects = [([], [])]
        causals = [(list(range(start, end+1)), group['token'].iloc[start:end+1].tolist()) for start, end in causal_spans]
        if not causals:
            causals = [([], [])]
        return effects, causals

    if 'actual' in group.columns.tolist():
        effects_actual, causals_actual = extract_spans('actual')
    else:
        effects_actual, causals_actual = None, None
    effects_pred, causals_pred = extract_spans('predicted')
    return effects_actual == effects_pred and causals_actual == causals_pred, effects_pred, causals_pred


def get_all_results():
    test_files = glob.glob(os.path.join('results', 'test-*.csv'))
    test_files = [f for f in test_files if 'catboost' not in os.path.basename(f) and 'stacked' not in os.path.basename(f)]
    for file in test_files:
        results = pd.read_csv(file, sep=';')
        perf = results.groupby('sentence').apply(extract_spans)
        print(f'Accuracy for {os.path.basename(file)}:', sum(perf)/len(perf))


def create_submission_file(test_file, test_src, output_file):
    results = pd.read_csv(test_file, sep=';')
    test_data = pd.read_csv(test_src, sep=';')
    test_data.columns = list(map(lambda _: _.strip(), test_data.columns))
    perf = results.groupby('sentence').apply(extract_spans)\
        .apply(lambda _: (' '.join(_[1][0][1]), ' '.join(_[2][0][1])))
    perf = pd.DataFrame(data=perf.tolist(), columns=['Effect', 'Cause'])
    perf['Text'] = test_data['Text']
    perf['Index'] = test_data['Index']
    perf = perf[['Index', 'Text', 'Cause', 'Effect']]
    perf.to_csv('task2.csv', index=None, sep=';')
    with zipfile.ZipFile(output_file, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write('task2.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-file", help="Test file")
    parser.add_argument("--source-file", help="Source file")
    parser.add_argument("--output-file", help="Output file")
    args = parser.parse_args()
    create_submission_file(args.test_file, args.source_file, args.output_file)
    # get_all_results()