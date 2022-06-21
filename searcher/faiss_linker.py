import json
from .data_processor import (
    save_vectorizer,
    load_vectorizer,
    load_dictionary_df_from_cache,
    save_dictionary_df,
    read_dictinary_file,
)
from util.log_util import get_logger
import logging
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from time import time
from tqdm import trange
import faiss
import numpy as np


logger = get_logger(name=__name__, log_file=None, log_level=logging.INFO, log_level_name='')
MB = 1048576


class EntityLinkerFaiss():
    def __init__(self, args) -> None:
        """ 
        Always uses lower case for entity names to let td-idf have lower dimensions.
        example:
            ner_entity = triple-negative breast cancer
            linked_entity = triple-negative receptor breast cancer
        """
        for arg, value in vars(args).items():
            setattr(self, arg, value)           
        self.args = args
        Path('cache').mkdir(exist_ok=True)
        self.vectorizer_file = Path('cache', f'{self.corpus_type}_tf_idf_vectorizer.pickle')
        self.vector_df_file = Path('cache', f'{self.corpus_type}_vector_df.h5')
        self.dictionary_df_file = Path('cache', f'{self.corpus_type}_dictionary_df.h5')
        self.milvus_configs = read_milvus_configs()
        self.vector_field_name = self.milvus_configs['field_names'][-1]
        self.mivlus_search_params = self.milvus_configs['search_params']
        self.extra_out_fields = self.extra_out_fields.split()
        self.columns_for_search = self.milvus_configs['columns_for_search'] + self.extra_out_fields
        self.collection_name = args.corpus_type
        self.dict_file_path = read_dictinary_file(args.corpus_type)
        self.tdidf_config = self.read_tdidf_config()
        logger.info(f'collection_name: {self.collection_name}')

    def insert_dictionary(self):
        dictionary_df = self.create_df_and_train_vectorizer()
        self._insert_data(dictionary_df)    
        logger.info('data is successfully inserted into milvus! End.')

    def create_df_and_train_vectorizer(self):
        """ the returned dictionary_df has the vector column which is not needed in search, but needed in insertion 
        """
        logger.info(f'Creates dictionary_df and train vectorizer')
        df = self.process_dictionary()
        logger.debug(f'dictionary_df shape: {df.shape}')
        df = self.tf_idf_fit(df)
        return df

    def read_tdidf_config(self):
        file = 'configs/TfidfVectorizer_configs.json'
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data[self.corpus_type]

    def tf_idf_fit(self, df):
        if self.cache_vector_df and self.vectorizer_file.exists() and self.vector_df_file.exists():
            logger.info(f'loads vector_df_file {self.vector_df_file}')
            df = load_dictionary_df_from_cache(self.vector_df_file)
        else:
            # for large dictionaries(17K items), min_df = 3; for small ones(< 1K items), min_df = 1
            ngram_range = (self.tdidf_config['ngram_range_left'], self.tdidf_config['ngram_range_right'])
            min_df = self.tdidf_config['min_df']
            analyzer = self.tdidf_config['analyzer']
            vectorizer = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, min_df=min_df)
            # vectorizer = BertTokenTfidfVectorizer()
            logger.info('starts training tf-idf')
            vectorizer.fit(df.name)
            logger.info('tf_idf-fit finishes and starts to add vector into dataframe!')
            df[self.vector_field_name] = df.name.apply(lambda x: vectorizer.transform([x]))
            self.drop_df_columns(df, self.milvus_configs['field_names'])            
            logger.info('dataframe finishes adding the vector field!')
            save_vectorizer(vectorizer, self.vectorizer_file)
            save_dictionary_df(df, self.vector_df_file)
        return df

    def process_dictionary(self):
        df = self.get_dataframe_from_dictionary()
        save_dictionary_df(df, self.dictionary_df_file)
        return df

    def get_dataframe_from_dictionary(self):
        if self.dictionary_type == 'csv':
            df = self.get_dataframe_from_dictionary_csv()
        if self.dictionary_type == 'tsv':
            df = self.get_dataframe_from_dictionary_csv(delimiter='\t')
        return df

    def get_dataframe_from_dictionary_csv(self, delimiter=','):        
        df = pd.read_csv(self.dict_file_path, delimiter=delimiter)
        logger.info(f'dictionary_file {self.dict_file_path} is read by pandas, top 5 items are:\n{df.head()}')
        df = df[~df['name'].isna()]
        df['uid'] = df.index
        # uid and name are two basic and must columns to keep
        self.drop_df_columns(df, kept_columns=self.columns_for_search)
        return df

    def _insert_data(self, df):
        logger.info(f'pandas for mivlus insert, top 5 items are:\n{df.head()}')
        dim_size = df.vector[0].toarray().shape[1]
        rows_number = df.shape[0]
        logger.info(f'vector_for_insert dim_size: {dim_size}')
        logger.info(f'df_for_insert shape: {df.shape}')
        logger.info(f'df_for_insert column names: {df.columns}')

        index = init_faiss(dim_size)
        
        target_vectors = np.array(df[self.vector_field_name].apply(lambda x: x.toarray()[0]).tolist()).astype(
            np.float32)
        logger.info('target_vectors ok')
        
        assert not index.is_trained
        logger.info(f'starts train ...')
        index.train(target_vectors)
        assert index.is_trained        
        logger.info(f'starts add ...')
        index.add(target_vectors)   
        index_save_path = f'cache/{self.collection_name}_faiss_index.dat'
        faiss.write_index(index, index_save_path)
        logger.info(f'save index at {index_save_path}')

    def drop_df_columns(self, df, kept_columns):
        dropped_columns = []
        for item in df:
            if item not in kept_columns:
                dropped_columns.append(item)
        if dropped_columns:
            df.drop(columns=dropped_columns, inplace=True)

    # def load_pretrained_data(self):
    #     try:
    #         df_for_search = load_dictionary_df_from_cache(self.dictionary_df_file)
    #         vectorizer = load_vectorizer(self.vectorizer_file)
    #     except Exception as identifier:
    #         logger.info('Loading failes, and please initialize milvus in advance!', exc_info=identifier)
    #         # raise(identifier)
    #         self.create_df_and_train_vectorizer()
    #         df_for_search = load_dictionary_df_from_cache(self.dictionary_df_file)            
    #         vectorizer = load_vectorizer(self.vectorizer_file)
    #     collection = init_milvus(self.collection_name, self.milvus_configs, 
    #         drop_collection=self.drop_collection)
    #     collection.load()
    #     logger.debug(f'collection {self.collection_name} is loaded')
    #     self.vectorizer, self.df_for_search, self.collection = vectorizer, df_for_search, collection

    def search_vector(self, entity_names):
        """ MUST firstly run load_pretrained_data, Must firstly convert to list 
        Most of time is used in milvus.search()
        """
        # logger.debug(f'tf-idf transform starts')
        vec = self.vectorizer.transform(entity_names).toarray()
        # logger.debug(f'milvus search starts')
        # t0 = time()
        search_results = self.collection.search(data=vec, anns_field=self.vector_field_name, 
            param=self.mivlus_search_params, limit=self.top_num)
        # t_milvus = time() - t0            
        results = []
        for search_item in search_results:
            # Only get the top 1 similarest linked name
            ids = search_item.ids[0]
            linked_name = self.df_for_search.name.iloc[ids]
            extra_info = {}
            for extra_field in self.extra_out_fields:
                extra_info[extra_field] = self.df_for_search[extra_field].iloc[ids]
            distance = search_item.distances[0]
            results.append((linked_name, distance, extra_info))
        
        # return results, t_milvus
        return results


def init_faiss(dim_size, nlist=1024):
    quantizer = faiss.IndexFlatL2(dim_size)  # the other index
    index = faiss.IndexIVFFlat(quantizer, dim_size, nlist, faiss.METRIC_L2)    
    return index

def read_milvus_configs(milvus_config_file = 'configs/milvus_configs.json'):
    with open(milvus_config_file, 'r', encoding='utf-8') as f:
        milvus_configs = json.load(f)
    return milvus_configs
