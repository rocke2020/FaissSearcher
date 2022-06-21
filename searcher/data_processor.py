"""  """
import pickle
import pandas as pd
from numpy import dot
from util.log_util import get_logger
import logging
import json


logger = get_logger(name=__name__, log_file=None, log_level=logging.DEBUG, log_level_name='')


def cosine_similarity(x1, x2):
    """ TfidfVectorizer auto normalized and so not need norm """
    return dot(x1, x2)


def save_vectorizer(vectorizer, vectorizer_file):
    with open(vectorizer_file, 'wb') as f:
        pickle.dump(vectorizer, f, protocol=5)
    logger.info(f'tfidf vectorizer is saved at {vectorizer_file}.')


def load_vectorizer(vectorizer_file):
    with open(vectorizer_file, 'rb') as f:
        vectorizer = pickle.load(f)
    logger.info(f'tfidf vectorizer is loaded from {vectorizer_file}.')
    return vectorizer


def save_dictionary_df(df, df_file):
    df.to_hdf(df_file, key='df', mode='w')
    logger.info(f'dictionary_file is saved at {df_file}.')


def load_dictionary_df_from_cache(df_file):
    df = pd.read_hdf(df_file, 'df')
    logger.info(f'dictionary_file {df_file} is read by pandas, top 5 items are:\n{df.head()}')
    logger.info(f'dictionary_file {df_file} shape:{df.shape}')
    return df


def read_dictinary_file(corpus_type):
    with open('configs/dictionary_files.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    dict_file = data[corpus_type]
    return dict_file