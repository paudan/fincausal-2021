#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
from collections import defaultdict
import numpy as np
import re
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tokenize.regexp import RegexpTokenizer
import pandas as pd


def clean_tokens(tokens, to_replace='[^\w\-\+\&\.\'\"]+'):
    lemma = WordNetLemmatizer()
    tokens = [re.sub(to_replace, ' ', token) for token in tokens]
    tokens = [lemma.lemmatize(token) for token in tokens]
    return tokens

def tokenize(mystr):
    tokenizer = RegexpTokenizer('[^ ]+')
    return tokenizer.tokenize(mystr)


def make_causal_input(lod, map_, silent=True):
    """
    :param lod: list of dictionaries
    :param map_: mapping of tags and values of interest, i.e. [('cause', 'C'), ('effect', 'E')]. The silent tags are by default taggerd as 'O'
    :return: dict of list of tuples for each sentence
    """
    dd = defaultdict(list)
    dd_ = []
    rx = re.compile(r"(\b[-']\b)|[\W_]")
    rxlist = [r'("\\)', r'(\\")']
    rx = re.compile('|'.join(rxlist))
    for i in range(len(lod)):
        line_ = lod[i]['sentence']
        line = re.sub(rx, '', line_)
        caus = lod[i]['cause']
        caus = re.sub(rx, '', caus)
        effe = lod[i]['effect']
        effe = re.sub(rx, '', effe)

        d = defaultdict(list)
        index = 0
        for idx, w in enumerate(word_tokenize(line)):
            index = line.find(w, index)
            if not index == -1:
                d[idx].append([w, index])
                index += len(w)
        d_ = defaultdict(list)
        for idx in d:
            d_[idx].append([tuple([d[idx][0][0], 'O']), d[idx][0][1]])

        init_e = line.find(effe)
        init_e = 0 if init_e == -1 else init_e
        init_c = line.find(caus)
        init_c = 0 if init_c == -1 else init_c

        for c, cl in enumerate(word_tokenize(caus)):
            init_c = line.find(cl, init_c)
            stop = line.find(cl, init_c) + len(cl)
            word = line[init_c:stop]
            for idx in d_:
                if int(init_c) == int(d_[idx][0][1]):
                    und_ = defaultdict(list)
                    und_[idx].append([tuple([word, 'C']), line.find(word, init_c)])
                    d_[idx] = und_[idx]

            init_c += len(cl)

        for e, el in enumerate(word_tokenize(effe)):
            init_e = line.find(el, init_e)
            stop = line.find(el, init_e) + len(el)
            word = line[init_e:stop]
            for idx in d_:
                if int(init_e) == int(d_[idx][0][1]):
                    und_ = defaultdict(list)
                    und_[idx].append([tuple([word, 'E']), line.find(word, init_e)])
                    d_[idx] = und_[idx]

            init_e += len(word)
        dd[i].append(d_)
    for dict_ in dd:
        dd_.append([item[0][0] for sub in [[j for j in i.values()] for i in lflatten(dd[dict_])] for item in sub])
    return dd_


def s2dict(lines, lot):
    d = defaultdict(list)
    for line_, tag_ in zip(lines, lot):
        d[tag_] = line_
    return d


def make_data(df):
    lodict_ = []
    for rows in df.itertuples():
        list_ = [rows[2], rows[3], rows[4]]
        map1 = ['sentence', 'cause', 'effect']
        dict_ = s2dict(list_, map1)
        lodict_.append(dict_)
    map_ = [('cause', 'C'), ('effect', 'E')]
    return zip(*[tuple(zip(*x)) for x in make_causal_input(lodict_, map_)])

def make_data2(df):
    lodict_ = []
    for rows in df.itertuples():
        list_ = [rows[2], rows[3], rows[4]]
        map1 = ['sentence', 'cause', 'effect']
        dict_ = s2dict(list_, map1)
        lodict_.append(dict_)
    map_ = [('cause', 'C'), ('effect', 'E')]
    import itertools
    return list(itertools.chain(*make_causal_input(lodict_, map_)))


def create_data_files(input_file_path, validation=False):
    df = pd.read_csv(input_file_path, delimiter='; ', engine='python', header=0)
    # Make train and test sets keeping multiple cause / effects blocks together.
    df['IdxSplit'] = df.Index.apply(lambda x: ''.join(x.split(".")[0:2]))
    df.set_index('IdxSplit', inplace=True)
    np.random.seed(0)
    testrows = np.random.choice(df.index.values, int(len(df) / 4))
    test_sents = df.loc[testrows].drop_duplicates(subset='Index')
    train_sents = df.drop(test_sents.index)
    if validation is True:
        validrows = np.random.choice(train_sents.index.values, int(len(train_sents) / 4))
        valid_sents = train_sents.loc[validrows]
        train_sents = df.drop(valid_sents.index)
        pairs = make_data2(valid_sents)
        pd.DataFrame(pairs).to_csv('valid_data.csv', sep=' ', index=None, header=False)
    pairs = make_data2(train_sents)
    pd.DataFrame(pairs).to_csv('train_data.csv', sep=' ', index=None, header=False)
    pairs = make_data2(test_sents)
    pd.DataFrame(pairs).to_csv('test_data.csv', sep=' ', index=None, header=False)


def create_data_files2(input_file_path, validation=False):

    def write_list(lst, outfile):
        with open(outfile, 'w') as f:
            for item in lst:
                f.write("%s\n" % item)

    df = pd.read_csv(input_file_path, delimiter='; ', engine='python', header=0)
    # Make train and test sets keeping multiple cause / effects blocks together.
    df['IdxSplit'] = df.Index.apply(lambda x: ''.join(x.split(".")[0:2]))
    df.set_index('IdxSplit', inplace=True)
    np.random.seed(0)
    testrows = np.random.choice(df.index.values, int(len(df) / 4))
    test_sents = df.loc[testrows].drop_duplicates(subset='Index')
    train_sents = df.drop(test_sents.index)
    if validation is True:
        validrows = np.random.choice(train_sents.index.values, int(len(train_sents) / 4))
        valid_sents = train_sents.loc[validrows]
        train_sents = train_sents.drop(valid_sents.index)
        sentences, tags = make_data(valid_sents)
        write_list(list(map(lambda x: ' '.join(x), sentences)), 'testa.words.txt')
        write_list(list(map(lambda x: ' '.join(x), tags)), 'testa.tags.txt')
    sentences, tags = make_data(train_sents)
    write_list(list(map(lambda x: ' '.join(x), sentences)), 'train.words.txt')
    write_list(list(map(lambda x: ' '.join(x), tags)), 'train.tags.txt')
    sentences, tags = make_data(test_sents)
    write_list(list(map(lambda x: ' '.join(x), sentences)), 'testb.words.txt')
    write_list(list(map(lambda x: ' '.join(x), tags)), 'testb.tags.txt')


def create_data_files3(input_file_path, test_file_path, validation=False):

    def write_list(lst, outfile):
        with open(outfile, 'w') as f:
            for item in lst:
                f.write("%s\n" % item)

    train_sents = pd.read_csv(input_file_path, delimiter='; ', engine='python', header=0)
    train_sents['IdxSplit'] = train_sents.Index.apply(lambda x: ''.join(x.split(".")[0:2]))
    train_sents.set_index('IdxSplit', inplace=True)
    test_sents = pd.read_csv(test_file_path, delimiter='; ', engine='python', header=0)
    test_sents['IdxSplit'] = test_sents.Index.apply(lambda x: ''.join(x.split(".")[0:2]))
    test_sents.set_index('IdxSplit', inplace=True)
    np.random.seed(0)
    if validation is True:
        validrows = np.random.choice(train_sents.index.values, int(len(train_sents) / 4))
        valid_sents = train_sents.loc[validrows]
        train_sents = train_sents.drop(valid_sents.index)
        sentences, tags = make_data(valid_sents)
        write_list(list(map(lambda x: ' '.join(x), sentences)), 'testa.words.txt')
        write_list(list(map(lambda x: ' '.join(x), tags)), 'testa.tags.txt')
    sentences, tags = make_data(train_sents)
    write_list(list(map(lambda x: ' '.join(x), sentences)), 'train.words.txt')
    write_list(list(map(lambda x: ' '.join(x), tags)), 'train.tags.txt')
    sentences = [' '.join([ word for idx, word in enumerate(word_tokenize(row[2]))]) for row in test_sents.itertuples()]
    write_list(sentences, 'testb.words.txt')
    # Just temp tags
    tags = [' '.join('O' for _ in word_tokenize(row[2])) for row in test_sents.itertuples()]
    write_list(tags, 'testb.tags.txt')


def evaluate(test_file_path, modelpath='', args_idx = 1):
    pred_file = '/mnt/DATA/python/tf_ner/models/chars_lstm_lstm_crf/results/score/testb.preds.txt'
    with open(pred_file, 'r') as f:
        predicted = []
        sent_data = []
        for line in f:
            line = line.strip()
            if len(line) > 0:
                items = line.split(' ')
                sent_data.append((items[0], items[1], items[2]))
            else:
                predicted.append(sent_data)
                sent_data = []
        if len(sent_data) > 0:
            predicted.append(sent_data)

    labels = {"C": 1, "E": 2, "O": 0}
    predictions = np.array([labels[pred] for sent in predicted for _, _, pred in sent])
    truths = np.array([labels[t] for sent in predicted for _, t, _ in sent])
    print(np.sum(truths == predictions) / len(truths))

    y_test = [[t for _, t, _ in sent] for sent in predicted]
    y_pred = [[pred for __, _, pred in sent] for sent in predicted]
    tokens_test = [[token  for token, _, _ in sent] for sent in predicted]

    ll = []
    for i, (pred, token) in enumerate(zip(y_pred, tokens_test)):
        l = defaultdict(list)
        for j, (y, word) in enumerate(zip(pred, token)):
            print(y, word)
            l[j] = (word, y)
        ll.append(l)

    nl = []
    for line, yt, yp in zip(ll, y_test, y_pred):
        d_ = defaultdict(list)
        d_["truth"] = yt
        d_["pred"] = yp
        d_["diverge"] = 0
        for k, v in line.items():
            d_[v[1]].append(''.join(v[0]))
        if d_["truth"] != d_["pred"]:
            d_["diverge"] = 1
        d_['Cause'] = ' '.join(el for el in d_['C'])
        cause_extend = len(d_['Cause']) + 1  # add 1 extra space at start
        d_[' Cause'] = d_['Cause'].rjust(cause_extend)
        d_['_'] = ' '.join(el for el in d_['_'])
        d_['Effect'] = ' '.join(el for el in d_['E'])
        effect_extend = len(d_['Effect']) + 1
        d_[' Effect'] = d_['Effect'].rjust(effect_extend)
        nl.append(d_)

    fieldn = sorted(list(set(k for d in nl for k in d)))
    with open(os.path.join(modelpath, ("controls_" + str(args_idx)) + ".csv"), "w+", encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldn, delimiter="~")
        writer.writeheader()
        for line in nl:
            writer.writerow(line)

    test = pd.read_csv(test_file_path, delimiter='; ', engine='python', header=0)
    test['IdxSplit'] = test.Index.apply(lambda x: ''.join(x.split(".")[0:2]))
    test.set_index('IdxSplit', inplace=True)

    tmp = pd.DataFrame.from_records(nl)[['Cause', 'Effect']].reset_index()
    idx = pd.DataFrame(test['Index']).reset_index()
    text = pd.DataFrame(test['Text']).reset_index()
    task2 = pd.concat([idx, text, tmp], axis=1)
    task2 = task2.drop(['index', 'IdxSplit'], axis=1)
    task2 = task2.sort_values('Index')
    test = test.sort_values('Index')
    task2.to_csv(os.path.join(modelpath, ("task2_eval_" + str(args_idx)) + ".csv"), sep = ';', index=False)
    test.to_csv(os.path.join(modelpath, ("task2_ref_" + str(args_idx)) + ".csv"), sep = ';', index=False)
