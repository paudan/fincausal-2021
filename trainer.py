import argparse
import torch
from flair.datasets.sequence_labeling import ColumnCorpus
from flair.embeddings import TransformerWordEmbeddings, ELMoEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

BERT_MODEL_DIR = '/mnt/DATA/data/nlp/FinBERT-FinVocab-Uncased'
CACHE_DIR = 'embeddings'
INPUT_PATH = 'practice'
MAX_LENGTH = 512
torch.set_default_tensor_type(torch.FloatTensor)

tagger_config = [
    { 'name': 'finbert-finvocab',
      'embeddings': TransformerWordEmbeddings(BERT_MODEL_DIR, cache_dir=CACHE_DIR)
    },
    { 'name': 'finbert-prosusai',
      'embeddings': TransformerWordEmbeddings('ProsusAI/finbert', cache_dir=CACHE_DIR)
    },
    { 'name': 'finbert-combo',
      'embeddings': TransformerWordEmbeddings('/mnt/DATA/data/nlp/FinBERT-Combo_128MSL-250K', cache_dir=CACHE_DIR)
    },
    { 'name': 'elmo-small',
      'embeddings': ELMoEmbeddings(model='small')
    },
    { 'name': 'roberta-model',
      'embeddings': TransformerWordEmbeddings('roberta-base', cache_dir=CACHE_DIR)
    },
    { 'name': 'bert-base',
    'embeddings': TransformerWordEmbeddings(cache_dir=CACHE_DIR)
    },
]

def train_tagger(input_path, train_file, test_file, dev_file):
    columns = {0 : 'text', 1 : 'ner'}
    corpus = ColumnCorpus(input_path, columns, train_file=train_file, test_file=test_file, dev_file=dev_file)
    tag_dictionary = corpus.make_tag_dictionary(tag_type='ner')
    for config in tagger_config:
        tagger = SequenceTagger(hidden_size=256, embeddings=config['embeddings'], tag_dictionary=tag_dictionary, tag_type='ner', use_crf=True)
        trainer = ModelTrainer(tagger, corpus, use_tensorboard=True)
        trainer.train(config['name'], learning_rate=0.1, mini_batch_size=32, max_epochs=100, embeddings_storage_mode='gpu')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", help="Dataset", default=INPUT_PATH)
    parser.add_argument("--train-file", help="Training file", default='train.txt')
    parser.add_argument("--test-file", help="Testing file", default=None)
    parser.add_argument("--dev-file", help="Development file", default='valid.txt')
    args = parser.parse_args()
    train_tagger(args.input_path, args.train_file, args.test_file, args.dev_file)
