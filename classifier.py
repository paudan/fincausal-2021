import itertools
import os
import argparse
import joblib
from typing import Dict
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from catboost.core import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

SEED = 42

def preprocess_catboost_data(features):
    features = features.drop(labels=['token', 'label', 'instance'], axis=1, errors='ignore')
    filtered = list(map(lambda x: 'word.lower' not in x, features.columns))
    features = features[features.columns[filtered]]
    boolean_columns = list(map(lambda x: any(s in x for s in {'istitle', 'isupper', 'isdigit'}), features.columns))
    boolean_features = features.columns[boolean_columns]
    features[boolean_features] = features[boolean_features].fillna(value=False).astype(int)
    features.fillna(value='NA', inplace=True)
    return features

def load_data(path):
    features = pd.read_csv(path, index_col=0)
    labels = features['label'].astype(str)
    return preprocess_catboost_data(features), labels

def catboost_classifier(train_data, labels):
    clf = CatBoostClassifier(verbose=False)
    clf.fit(train_data, labels)
    return clf

def random_forest_classifier(train_data, labels):
    clf = RandomForestClassifier(n_estimators=200)
    clf.fit(train_data, labels)
    return clf

def voting_classifier(X_train, y_train):
    major = X_train[y_train==0]
    minor = X_train[y_train==1]
    kf = KFold(n_splits=10, random_state=SEED, shuffle=True)
    classifiers = []
    i = 1
    for ind, train_index in kf.split(major):
        print(f'Training classifier for subsample {i}')
        df_sub = np.vstack([major[train_index], minor])
        y_sub = [0] * len(train_index) + [1] * minor.shape[0]
        clf = catboost_classifier(df_sub, y_sub)
        classifiers.append((f'clf{i}', clf))
        i += 1
    clf = VotingClassifier(estimators=classifiers, voting='soft')
    clf.fit(X_train, y_train)
    return clf

def weighted_catboost_classifier(train_data, labels):
    classes=np.unique(labels)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels)
    class_weights = dict(zip(classes, weights))
    clf = CatBoostClassifier(verbose=False, class_weights=class_weights)
    clf.fit(train_data, labels)
    return clf

def plain_catboost_classifier(train_data, y_train, weighted=False):
    X_train = train_data.copy()
    X_train.fillna(value='NA', inplace=True)
    categorical_features = X_train.columns[X_train.dtypes == object]
    X_train[categorical_features] = X_train[categorical_features].astype('category')
    params = {'learning_rate': [0.03, 0.1],
            'depth': [4, 6, 10],
            'l2_leaf_reg': [1, 3, 5, 7, 9]}
    kwargs = {}
    if weighted is True:
        classes=np.unique(y_train)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
        class_weights = dict(zip(classes, weights))
        kwargs = {'class_weights': class_weights}
    cb = CatBoostClassifier(verbose=True, cat_features=train_data.columns.tolist(), **kwargs)
    cb.fit(X_train, y_train)
    # cb.randomized_search(params, X=X_train, y=y_train)
    # cb.grid_search(params, X=X_train, y=y_train)
    return cb

def preprocess_data(X_train, y_train):
    train_data = X_train.copy()
    labels_train = y_train.copy()
    label_encoder = LabelEncoder().fit(labels_train)
    labels = label_encoder.transform(labels_train)
    boolean_features = train_data.columns[train_data.dtypes == bool]
    train_data[boolean_features] = train_data[boolean_features].astype(int)
    categorical_features_mask = train_data.dtypes == object
    categorical_features = train_data.columns[categorical_features_mask]

    def encode_column(train_data, col):
        train_data.loc[pd.isnull(train_data[col]), col] = 'N/A'
        encoder = LabelEncoder()
        train_data[col] = encoder.fit_transform(train_data[col])
        return encoder

    column_encoders = dict()
    ohe = None
    if len(categorical_features) > 0:
        column_encoders = {x: encode_column(train_data, x) for x in categorical_features}
        ohe = ColumnTransformer([('encoder', OneHotEncoder(handle_unknown='ignore'), categorical_features_mask)],
                                remainder='passthrough', sparse_threshold=0)
        train_data = ohe.fit_transform(train_data)
    return train_data, labels, column_encoders, ohe, label_encoder

def preprocess_test_data(X_test, y_test, column_encoders: Dict[str, LabelEncoder], ohe: ColumnTransformer, label_encoder: LabelEncoder):
    def apply_encoder(encoder, data):
        enc_dict = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
        return data.apply(lambda x: enc_dict.get(x, 999999))  # Encode "unseen" values
    test_data = X_test.copy()
    labels_test = label_encoder.transform(y_test)
    categorical_features = column_encoders.keys()
    if len(categorical_features) > 0:
        for x in categorical_features:
            test_data[x] = apply_encoder(column_encoders[x], test_data[x])
        test_data = ohe.transform(test_data)
    return test_data, labels_test


def test_classifier(clf, X_test, y_test, column_encoders: Dict[str, LabelEncoder], ohe: ColumnTransformer, label_encoder: LabelEncoder):
    test_data, labels_test = preprocess_test_data(X_test, y_test, column_encoders, ohe, label_encoder)
    results = clf.predict(test_data).astype(int)
    results = pd.DataFrame({'predicted': label_encoder.inverse_transform(results),
                            'original': label_encoder.inverse_transform(labels_test)})
    try:
        probs = pd.DataFrame(clf.predict_proba(test_data), columns=['Class_{}'.format(i) for i in set(labels_test)])
        results = pd.concat([results.reset_index(drop=True), probs.reset_index(drop=True)], axis='columns')
    except AttributeError:
        pass
    return results.reset_index(drop=True)


def train_classifier(X_train, y_train, output_dir: str='.', X_test=None, y_test=None, classifier=catboost_classifier):
    train_data, labels, column_encoders, ohe, label_encoder = preprocess_data(X_train, y_train)
    categorical_features = list(column_encoders.keys())
    clf = classifier(train_data, labels)
    if output_dir is not None:
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        joblib.dump(clf, os.path.join(output_dir, 'model.joblib'))
        joblib.dump((categorical_features, label_encoder, column_encoders, ohe),
                    os.path.join(output_dir, 'preprocess.joblib'))
    results = None
    if X_test is not None:
        results = test_classifier(clf, X_test, y_test, column_encoders, ohe, label_encoder)
    return clf, results, ohe.get_feature_names()

def predict_plain_catboost(model_path, input_file, output_file):
    model = joblib.load(model_path)
    test_data = pd.read_csv(input_file, sep=';').reset_index(drop=True)
    features = preprocess_catboost_data(test_data)
    categorical_features = features.columns[features.dtypes == object]
    features[categorical_features] = features[categorical_features].astype('category')
    results = model.predict(features)
    if 'label' in test_data.columns.tolist():
        labels_test = test_data['label'].astype(str)
        results = pd.DataFrame({'predicted': list(itertools.chain(*results)),
                                'original': labels_test,
                                'sentence': test_data['instance'],
                                'token': test_data['token']})
        try:
            probs = pd.DataFrame(model.predict_proba(features), columns=['Class_{}'.format(i) for i in set(labels_test)])
            results = pd.concat([results.reset_index(drop=True), probs.reset_index(drop=True)], axis='columns')
        except AttributeError:
            pass
    else:
         results = pd.DataFrame({'predicted': list(itertools.chain(*results)),
                                 'sentence': test_data['instance'],
                                 'token': test_data['token']})
    results.reset_index(drop=True).to_csv(output_file, index=False, sep=';')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", help="Training data file path")
    parser.add_argument("--test-file", help="Testing data file path")
    args = parser.parse_args()
    X_train, y_train = load_data(args.train_file)
    X_test, y_test = load_data(args.test_file)
    cb = plain_catboost_classifier(X_train, y_train)
    output_dir = 'classifier-catboost'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    model_path = os.path.join(output_dir, 'model.joblib')
    joblib.dump(cb, model_path)

    test_data = X_test.copy()
    test_data.fillna(value='NA', inplace=True)
    categorical_features = test_data.columns[test_data.dtypes == object]
    test_data[categorical_features] = test_data[categorical_features].astype('category')
    predicted = cb.predict(test_data)
    print(classification_report(predicted, y_test))
    print(cb.get_feature_importance(prettified=True))

    predict_plain_catboost(model_path, args.train_file, os.path.join('results', 'train-catboost.csv'))
    predict_plain_catboost(model_path, args.test_file, os.path.join('results', 'test-catboost.csv'))