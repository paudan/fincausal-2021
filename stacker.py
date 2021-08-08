import os
import itertools
from collections import Counter
import joblib
import pandas as pd
from catboost.core import CatBoostClassifier
from sklearn.metrics import accuracy_score

output_dir = 'catboost-stacked'

def train_stacked():
    catboost = pd.read_csv(os.path.join('results', 'train-catboost.csv'), sep=';')
    elmo = pd.read_csv(os.path.join('results', 'train-elmo-small.csv'), sep=';')
    bert = pd.read_csv(os.path.join('results', 'train-bert-base.csv'), sep=';')
    finbert1 = pd.read_csv(os.path.join('results', 'train-finbert-prosusai.csv'), sep=';')
    finbert2 = pd.read_csv(os.path.join('results', 'train-finbert-combo.csv'), sep=';')
    roberta = pd.read_csv(os.path.join('results', 'train-roberta-model.csv'), sep=';')
    data = pd.DataFrame([catboost['predicted'], elmo['predicted'], bert['predicted'],
                         finbert1['predicted'], finbert2['predicted'], roberta['predicted']]).T
    data.columns = ['catboost', 'elmo', 'bert', 'finbert1', 'finbert2', 'roberta']
    labels = catboost['original']
    data = data.astype('category')
    params = {'learning_rate': [0.03, 0.1],
            'depth': [4, 6, 10],
            'l2_leaf_reg': [1, 3, 5, 7, 9]}
    cb = CatBoostClassifier(verbose=True, cat_features=data.columns.tolist())
    # cb.grid_search(params, X=data, y=labels)
    cb.fit(data, labels)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    joblib.dump(cb, os.path.join(output_dir, 'model.joblib'))

def test_majority_voting(prefix: str='test', output_accuracies: bool=False):
    catboost = pd.read_csv(os.path.join('results', f'{prefix}-catboost.csv'), sep=';')
    elmo = pd.read_csv(os.path.join('results', f'{prefix}-elmo-small.csv'), sep=';')
    bert = pd.read_csv(os.path.join('results', f'{prefix}-bert-base.csv'), sep=';')
    finbert1 = pd.read_csv(os.path.join('results', f'{prefix}-finbert-prosusai.csv'), sep=';')
    finbert2 = pd.read_csv(os.path.join('results', f'{prefix}-finbert-combo.csv'), sep=';')
    roberta = pd.read_csv(os.path.join('results', f'{prefix}-roberta-model.csv'), sep=';')
    data = pd.DataFrame([catboost['predicted'], elmo['predicted'], bert['predicted'],
                         finbert1['predicted'], finbert2['predicted'], roberta['predicted']]).T
    data.columns = ['catboost', 'elmo', 'bert', 'finbert1', 'finbert2', 'roberta']
    results = pd.DataFrame({
        'sentence': bert['sentence'],
        'token': bert['token'],
        'predicted': data.apply(lambda x: Counter(x.tolist()).most_common(1)[0][0], axis=1)
    })
    if 'actual' in bert.columns.tolist():
        results['actual'] = bert['actual']
    results = pd.concat([results, data], axis=1)
    results.reset_index(drop=True).to_csv(os.path.join('results', f'{prefix}-voting.csv'), index=False, sep=';')
    if not output_accuracies is True:
        return
    print('CatBoost accuracy:', accuracy_score(catboost['original'], catboost['predicted']))
    print('BERT accuracy:', accuracy_score(bert['actual'], bert['predicted']))
    print('ELMO accuracy:', accuracy_score(elmo['actual'], elmo['predicted']))
    print('FinBERT1 accuracy:', accuracy_score(finbert1['actual'], finbert1['predicted']))
    print('FinBERT2 accuracy:', accuracy_score(finbert2['actual'], finbert2['predicted']))
    print('RoBERTa accuracy:', accuracy_score(roberta['actual'], roberta['predicted']))
    print('Voted classifier accuracy:', accuracy_score(results['actual'], results['predicted']))


def test_stacked(prefix: str='test', output_accuracies: bool=False):
    catboost = pd.read_csv(os.path.join('results', f'{prefix}-catboost.csv'), sep=';')
    elmo = pd.read_csv(os.path.join('results', f'{prefix}-elmo-small.csv'), sep=';')
    bert = pd.read_csv(os.path.join('results', f'{prefix}-bert-base.csv'), sep=';')
    finbert1 = pd.read_csv(os.path.join('results', f'{prefix}-finbert-prosusai.csv'), sep=';')
    finbert2 = pd.read_csv(os.path.join('results', f'{prefix}-finbert-combo.csv'), sep=';')
    roberta = pd.read_csv(os.path.join('results', f'{prefix}-roberta-model.csv'), sep=';')
    data = pd.DataFrame([catboost['predicted'], elmo['predicted'], bert['predicted'],
                         finbert1['predicted'], finbert2['predicted'], roberta['predicted']]).T
    data.columns = ['catboost', 'elmo', 'bert', 'finbert1', 'finbert2', 'roberta']
    data = data.astype('category')
    model = joblib.load(os.path.join(output_dir, 'model.joblib'))
    results = model.predict(data)
    if 'original' in catboost.columns:
        results = pd.DataFrame({'predicted': list(itertools.chain(*results)),
                                'original': catboost['original'],
                                'sentence': catboost['sentence'],
                                'token': catboost['token']})
        try:
            probs = pd.DataFrame(model.predict_proba(data), columns=['Class_{}'.format(i) for i in set(catboost['original'])])
            results = pd.concat([results.reset_index(drop=True), probs.reset_index(drop=True)], axis='columns')
        except AttributeError:
            pass
    else:
         results = pd.DataFrame({'predicted': list(itertools.chain(*results)),
                                 'sentence': catboost['sentence'],
                                 'token': catboost['token']})
    results.reset_index(drop=True).to_csv(os.path.join('results', f'{prefix}-stacked.csv'), index=False, sep=';')
    if not output_accuracies is True:
        return
    print('CatBoost accuracy:', accuracy_score(catboost['original'], catboost['predicted']))
    print('BERT accuracy:', accuracy_score(bert['actual'], bert['predicted']))
    print('ELMO accuracy:', accuracy_score(elmo['actual'], elmo['predicted']))
    print('FinBERT1 accuracy:', accuracy_score(finbert1['actual'], finbert1['predicted']))
    print('FinBERT2 accuracy:', accuracy_score(finbert2['actual'], finbert2['predicted']))
    print('RoBERTa accuracy:', accuracy_score(roberta['actual'], roberta['predicted']))
    print('Stacked classifier accuracy:', accuracy_score(results['original'], results['predicted']))


if __name__ == '__main__':
#    train_stacked()
    test_stacked('tag')
    test_majority_voting('tag')

