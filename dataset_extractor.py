import os
import re
import argparse
from collections import defaultdict
import itertools
import logging
from funcy import lflatten
from tqdm import tqdm
import pandas as pd
import torch
import stanza
import nltk

logging.basicConfig(level=logging.DEBUG, format='%(process)d-%(levelname)s-%(message)s')
NLTK_PATH = '/mnt/DATA/data/nltk'
nltk.data.path.append(NLTK_PATH)
STANZA_DIR = os.path.join('/mnt/DATA/Darbas/KTU/code/m2m-nlp-experiment', 'stanza_resources')
# stanza.download('en', model_dir=STANZA_DIR)
torch.set_default_tensor_type(torch.FloatTensor)
tagger = stanza.Pipeline('en', dir=STANZA_DIR, processors='tokenize,pos,lemma,depparse')

# rx = re.compile(r"(\b[-']\b)|[\W_]")
rxlist = [r'("\\)', r'(\\")']
rx = re.compile('|'.join(rxlist))

def s2dict(lines, lot):
    d = defaultdict(list)
    for line_, tag_ in zip(lines, lot):
        d[tag_] = line_
    return d

def tokenize(line):
    # doc = tagger(line)
    # w = [[tok.text for tok in sentence.words] for sentence in doc.sentences]
    # return list(itertools.chain(*w))
    return nltk.word_tokenize(line)

def make_causal_input(lod, use_iobe=False):
    dd = defaultdict(list)
    dd_ = []
    for i in tqdm(range(len(lod))):
        line_ = lod[i]['sentence']
        line = re.sub(rx, '', line_)
        caus = lod[i]['cause']
        caus = re.sub(rx, '', caus)
        effe = lod[i]['effect']
        effe = re.sub(rx, '', effe)
        d = defaultdict(list)
        index = 0
        for idx, w in enumerate(tokenize(line)):
            index = line.find(w, index)
            if not index == -1:
                d[idx].append([w, index])
                index += len(w)

        d_ = defaultdict(list)
        for idx in d:
            d_[idx].append([tuple([d[idx][0][0], '_']), d[idx][0][1]])

        init_e = line.find(effe)
        init_e = 0 if init_e == -1 else init_e
        init_c = line.find(caus)
        init_c = 0 if init_c == -1 else init_c
        entry = tokenize(caus)
        for c, cl in enumerate(entry):
            init_c = line.find(cl, init_c)
            stop = line.find(cl, init_c) + len(cl)
            word = line[init_c:stop]
            for idx in d_:
                if int(init_c) == int(d_[idx][0][1]):
                    und_ = defaultdict(list)
                    tag = 'C'
                    if use_iobe is True:
                        if c == 0:
                            tag = 'B-C'
                        elif c == len(entry) - 1:
                            tag = 'E-C'
                    und_[idx].append([tuple([word, tag]), line.find(word, init_c)])
                    d_[idx] = und_[idx]
            init_c += len(cl)

        entry = tokenize(effe)
        for e, el in enumerate(entry):
            init_e = line.find(el, init_e)
            stop = line.find(el, init_e) + len(el)
            word = line[init_e:stop]
            for idx in d_:
                if int(init_e) == int(d_[idx][0][1]):
                    und_ = defaultdict(list)
                    tag = 'E'
                    if use_iobe is True:
                        if e == 0:
                            tag = 'B-E'
                        elif e == len(entry) - 1:
                            tag = 'E-E'
                    und_[idx].append([tuple([word, tag]), line.find(word, init_e)])
                    d_[idx] = und_[idx]
            init_e += len(word)

        dd[i].append(d_)
    for dict_ in dd:
        dd_.append([item[0][0] for sub in [[j for j in i.values()] for i in lflatten(dd[dict_])] for item in sub])
    return dd_


def tagPOS(lod, tagger):
    su_pos = []
    for i in tqdm(range(len(lod))):
        line_ = lod[i]['sentence']
        text = re.sub(rx, '', line_)
        doc = tagger(text)
        pos_ = [[{'Token': tok.text,
                  'dep': tok.deprel, 'pos': tok.upos,
                  'dep.head': sentence.words[tok.head-1].text if tok.head > 0 else "root",
                  'dep.head.pos': sentence.words[tok.head-1].pos if tok.head > 0 else None,
                  'instance': lod[i]['index']}
                 for tok in sentence.words]
                for sentence in doc.sentences]
        su_pos.append(list(itertools.chain(*pos_)))
    return su_pos

def tagPOS_nltk(lod):
    su_pos = []
    for i in tqdm(range(len(lod))):
        line_ = lod[i]['sentence']
        text = re.sub(rx, '', line_)
        doc = tagger(text)
        tokens = nltk.word_tokenize(text)
        su_pos.append([{
            'Token': tok,
            'instance': lod[i]['index']
        } for tok in tokens])
    return su_pos

def word2features(sent, i, wsize=3):
    word = sent[i].get('Token')
    features = {
        'instance': sent[i].get('instance'),
        'token': sent[i].get('Token'),
        'word.lower': word.lower(),
        'word.isupper': word.isupper(),
        'word.istitle': word.istitle(),
        'word.isdigit': word.isdigit(),
        'postag':sent[i].get('pos'),
        'dep':sent[i].get('dep'),
        'dep.head.pos':sent[i].get('dep.head.pos')
    }
    for k in range(1, wsize + 1):
        if i > k-1:
            word1 = sent[i-k].get('Token')
            features.update({
                f'-{k}:word.lower': word1.lower(),
                f'-{k}:word.istitle': word1.istitle(),
                f'-{k}:word.isupper': word1.isupper(),
                f'-{k}:word.isdigit': word1.isdigit(),
                f'-{k}:postag': sent[i-k].get('pos'),
                f'-{k}:dep': sent[i-k].get('dep'),
                f'-{k}:dep.head.pos': sent[i-k].get('dep.head.pos')
            })

        if i < len(sent)-k:
            word1 = sent[i+k].get('Token')
            features.update({
                f'+{k}:word.lower': word1.lower(),
                f'+{k}:word.istitle': word1.istitle(),
                f'+{k}:word.isupper': word1.isupper(),
                f'+{k}:word.isdigit': word1.isdigit(),
                f'+{k}:postag': sent[i+k].get('pos'),
                f'-{k}:dep': sent[i+k].get('dep'),
                f'-{k}:dep.head.pos': sent[i+k].get('dep.head.pos')
            })
    return features

# A function for extracting features in documents
def extract_features(doc):
    return [word2features(doc, i) for i in range(len(doc))]

# A function fo generating the list of labels for each document: TOKEN, POS, LABEL
def get_multi_labels(doc):
    return [label for (token, postag, label) in doc]


def process_document(input_file, output_file, use_iobe=False):
    df = pd.read_csv(input_file, delimiter=';', header=0)
    lodict_ = []
    if df.columns.size >= 4:
        for rows in df.itertuples():
            list_ = [rows[1], rows[2], rows[3], rows[4]]
            map1 = ['index', 'sentence', 'cause', 'effect']
            dict_ = s2dict(list_, map1)
            lodict_.append(dict_)
        hometags = make_causal_input(lodict_, use_iobe=use_iobe)
        # postags = tagPOS(lodict_, tagger)
        postags = tagPOS_nltk(lodict_)
        for i, (j, k) in enumerate(zip(hometags, postags)):
            if len(j) != len(k):
                print('POS alignment warning, ', i)
        tags = list(itertools.chain(*hometags))
        tags = pd.DataFrame(tags, columns=['token', 'label'])
        features = list(map(lambda entry: list(map(lambda _: word2features(entry, _), range(len(entry)))), postags))
        features = pd.DataFrame(list(itertools.chain(*features)))
        features = features.drop(labels='token', axis=1)
        data = pd.concat([tags, features], axis=1).reset_index(drop=True)
    elif df.columns.size == 2:
        for rows in df.itertuples():
            list_ = [rows[1], rows[2]]
            map1 = ['index', 'sentence']
            dict_ = s2dict(list_, map1)
            lodict_.append(dict_)
        postags = tagPOS(lodict_, tagger)
        features = list(map(lambda entry: list(map(lambda _: word2features(entry, _), range(len(entry)))), postags))
        features = pd.DataFrame(list(itertools.chain(*features)))
        data = features.reset_index(drop=True)
    else:
        raise Exception('Invalid file format')
    data.to_csv(output_file, sep=';', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", help="Input file path")
    parser.add_argument("--output-path", help="Output file path")
    parser.add_argument("--use-iobe", help="Use IOBE tagging scheme (set begin and end tags)", dest='use_iobe', action='store_true')
    parser.set_defaults(use_iobe=False)
    args = parser.parse_args()
    process_document(args.input_path, args.output_path, use_iobe=args.use_iobe)

