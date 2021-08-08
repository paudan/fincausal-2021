import os
import math
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


def get_split_indices(data: pd.DataFrame, valid_size: float=0.4, use_test_set: bool=True):
    ind_train, ind_test = train_test_split(data['instance'].unique(), test_size=valid_size)
    if use_test_set is True:
        ind_valid, ind_test = train_test_split(ind_test, test_size=0.5)
    else:
        ind_valid, ind_test = ind_test, None
    return ind_train, ind_valid, ind_test


def split_training_dataset(data: pd.DataFrame, output_dir: str, ind_train, ind_valid, ind_test):

    def create_file(df: pd.DataFrame, output_path: str):
        outputs = df.groupby(by='instance')[['token', 'label']].apply(lambda x: x.values.tolist())
        with open(output_path, 'w', encoding='utf-8') as f:
            for row in outputs.values:
                for item in row:
                    f.writelines([str(item[0]), ' ', item[1], '\n'])
                f.write("\n")

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    train_data = data[data['instance'].isin(ind_train)][['token', 'label', 'instance']]
    valid_data = data[data['instance'].isin(ind_valid)][['token', 'label', 'instance']]
    train_data['label'] = train_data['label'].replace('_', 'O')
    valid_data['label'] = valid_data['label'].replace('_', 'O')
    create_file(train_data, os.path.join(output_dir, 'train.txt'))
    create_file(valid_data, os.path.join(output_dir, 'valid.txt'))
    if ind_test is not None:
        test_data = data[data['instance'].isin(ind_test)][['token', 'label', 'instance']]
        test_data['label'] = test_data['label'].replace('_', 'O')
        create_file(test_data, os.path.join(output_dir, 'test.txt'))


def split_testing_dataset(data: pd.DataFrame, output_dir: str):

    def create_file(df: pd.DataFrame, output_path: str):
        outputs = df.groupby(by='instance')[['token']].apply(lambda x: x.values.tolist())
        with open(output_path, 'w', encoding='utf-8') as f:
            for row in outputs.values:
                for item in row:
                    f.writelines([str(item[0]), '\n'])
                f.write("\n")

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    create_file(data[['token', 'instance']], os.path.join(output_dir, 'test.txt'))


def split_features(data: pd.DataFrame, output_dir: str, ind_train, ind_valid, ind_test, suffix=""):
    train_data = data[data['instance'].isin(ind_train)]
    train_data['label'] = train_data['label'].replace('_', 'O')
    train_data.to_csv(os.path.join(output_dir, suffix + 'train.csv'), index=False)
    valid_data = data[data['instance'].isin(ind_valid)]
    valid_data['label'] = valid_data['label'].replace('_', 'O')
    valid_data.to_csv(os.path.join(output_dir, suffix + 'valid.csv'), index=False)
    if ind_test is not None:
        test_data = data[data['instance'].isin(ind_test)]
        test_data['label'] = test_data['label'].replace('_', 'O')
        test_data.to_csv(os.path.join(output_dir, suffix + 'test.csv'), index=False)


def create_datasets(data: pd.DataFrame, corpus_output_dir: str, features_dir: str, test_size: float=0.4, use_test_set: bool=True):
    if math.isclose(test_size, 1.0):
        split_testing_dataset(data, corpus_output_dir)
    else:
        ind_train, ind_valid, ind_test = get_split_indices(data, valid_size=test_size, use_test_set=use_test_set)
        split_training_dataset(data, corpus_output_dir, ind_train, ind_valid, ind_test)
        split_features(data, features_dir, ind_train, ind_valid, ind_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", help="Input file")
    parser.add_argument("--output-path", help="Output directory")
    parser.add_argument("--test-size", help="Validation/testing data percentage", default=0.4, type=float)
    parser.add_argument("--use-test-set", help="Use testing set for evaluation", dest='test_set', action='store_true')
    parser.set_defaults(test_set=False)
    args = parser.parse_args()
    data = pd.read_csv(args.input_path, sep=';')
    create_datasets(data, args.output_path, args.output_path, args.test_size, args.test_set)