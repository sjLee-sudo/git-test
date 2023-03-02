import re
import time
from elasticsearch.exceptions import TransportError
import pytz
import copy
import datetime
import pandas as pd
import numpy as np
from os import cpu_count
from functools import partial
import elasticsearch
import json
from elasticsearch import Elasticsearch, helpers, RequestError
from elasticsearch.client import CatClient
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
import timeit

if __name__ != '__main__':
    from config.constants import DATABASES, DOCUMENT_TABLE_NAME,TERM_TABLE_NAME, TERM_TYPES, DOCUMENT_DB_NAMES
    from documents.models import Document
    from django.db import connections
    from config.constants import MAX_CLAUSE_COUNT

DB_ENGINE = {
    'MYSQL': 'mysql',
    'DJANGO.DB.BACKENDS.MYSQL': 'mysql'
}

def return_timestamp():
    return (
        datetime.datetime.now(pytz.timezone("Asia/Seoul")).strftime(
            "%Y-%m-%dT%H:%M:%S.%f"
        )[:-3]
        + "Z"
    )


def create_sqlalchemy_engine(access_info):
    charset = 'utf8'
    if 'CHARSET' in access_info:
        charset = access_info['CHARSET']
    db_url = f"{DB_ENGINE[access_info['ENGINE'].upper()]}://{access_info['USER']}:{access_info['PASSWORD']}@{access_info['HOST']}:{access_info['PORT']}/{access_info['NAME']}?charset={charset}"
    return create_engine(db_url)

def default_processing_source_df(source_df):
    if 'title' in source_df.columns:
        source_df['title'] = source_df['title'].fillna('')
    if 'content' in source_df.columns:
        source_df['content'] = source_df['content'].fillna('')
    if 'term' in source_df.columns:
        source_df['term'] = source_df['term'].fillna('')
        source_df = source_df.drop_duplicates(['term'])
    return source_df

def post_processing_keit_data(source_df):
    if 'doc_subclass' in source_df.columns:
        source_df['doc_subclass'] = source_df['doc_subclass'].fillna('')
    if not 'doc_section' in source_df.columns:
        source_df['doc_section'] = source_df['doc_subclass'].map(
            lambda x: str(x)[:1])
    if not 'doc_class' in source_df.columns:
        source_df['doc_class'] = source_df['doc_subclass'].map(
            lambda x: str(x)[:3])
    return source_df


def get_dataframe_from_source_db_info_dict(source_db_info):
    '''
    source_db_info format
    {
        'access_info': {
            'USER': 'USER',
            'PASSWORD': PASSWORD,
            'HOST': HOST,
            'PORT': PORT,
            'NAME':'NAME',
            'ENGINE': 'MYSQL',
        },
        'table_name': 'table_name', 
        'columns': [
            'column_name1',
            'column_name2',
            ...
        ]
    }
    '''
    source_df = pd.DataFrame()
    if 'access_info' in source_db_info and 'table_name' in source_db_info and 'columns' in source_db_info:
        access_info = source_db_info['access_info']
        table_name = source_db_info['table_name']
        columns = source_db_info['columns']
        charset = 'utf8'
        if 'CHARSET' in access_info:
            charset = access_info['CHARSET']
        if isinstance(columns, list):
            column_names = ','.join(map(str, columns))
        else:
            column_names = '*'
        sql_query = f'SELECT {column_names} FROM {table_name};'

        db_engine = create_sqlalchemy_engine(access_info)
        with db_engine.connect() as db_conn:
            source_df = pd.read_sql(sql_query, con=db_conn)
    return source_df

# 멀티프로세스 실행 시 클래스 내 instance attributes 전달 시 thread_lock 문제로 에러 발생
# 외부 함수로 전환 후 필요값 인자로 전달하여 해결 -> 효율적인 구조로 변경 필요
def insert_format(index_name, df):
    data = []
    # 임시 처리
    # if "doc_section" in df.columns:
    #     df["doc_section"] = df["doc_section"].map(lambda x:  "UK" if x == "" else x)
    # if "doc_class" in df.columns:
    #     df["doc_class"] = df["doc_class"].map(lambda x:  "UK99" if x == "" else x)
    # if "doc_subclass" in df.columns:
    #     df["doc_subclass"] = df["doc_subclass"].map(lambda x:  "UK9999" if x == "" else x)    

    source_data = df.to_dict('records')
    for _source in source_data:
        insert_format = {'_index': index_name, '_source': {}}
        for k,v in _source.items():
            if k == 'doc_id':
                insert_format.update({'_id':v})
            if k == 'doc_section':
                insert_format.update({'_routing':v})
            insert_format['_source'].update({k:v})
        insert_format['_source'].update({'timestamp': datetime.datetime.now(
                pytz.timezone('Asia/Seoul')
            ).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]
            + 'Z'})
        data.append(insert_format)
    return data

def insert_elastic_data(index_name, es_info, df, max_retries=3):
    es = Elasticsearch(host=es_info['HOST'], port=es_info['PORT'], http_auth=(
            es_info['USER'],es_info['PASSWORD']), timeout=360, max_retries=max_retries, retry_on_timeout=True)
    err_idx = False
    err_ids = []
    for _try in range(1,max_retries+1):
        if err_idx:
            df = df[df['doc_id'].isin(err_ids)]
            err_ids = []
        data = insert_format(index_name,df)
        insert_result = helpers.streaming_bulk(client=es, actions=data, max_retries=max_retries, initial_backoff=3, max_backoff=600, yield_ok=False, raise_on_error=False)
        err_result = list(insert_result)
        if len(err_result)>0:
            print(len(err_result), f'not inserted, retry count: {_try}')
            err_idx = True
            for err in err_result:
                err_ids.append(err[1]['index']['_id'])
            time.sleep(10)
        else:
            break
    if len(err_result) > (df.shape[0])*0.1:
        raise Exception("non-successful insert data over 10% of data. stop migration. Error doc_ids: ",err_result)
    print('insert_elastic_data finished')
    return err_result
    
# index 생성 후 오류 항목들 재시도 위한 함수
def insert_elastic_controller(index_name, es_info, max_retries=3):
    for _try in range(0,max_retries):
        res = insert_elastic_data(index_name,es_info,max_retries)
        if len(res)==0:
            break

class BaseMigrator:

    def __init__(self, multiprocess_mode=False, multiprocess_worker_num=int(cpu_count()*0.2)):
        self.multiprocess_mode = multiprocess_mode
        self.multiprocess_worker_num = multiprocess_worker_num

    def _make_source_df_by_migration_info(self, migration_info):
        reversed_column_mapping = {}
        source_column_name_list = []
        source_db_info = migration_info
        if 'access_info' in migration_info and 'table_name' in migration_info and 'target_source_column_mapping' in migration_info:
            for target_col, src_col in migration_info['target_source_column_mapping'].items():
                if src_col != '' or len(src_col) != 0:
                    # 필요없는 컬럼 제외 및 dataframe 컬럼명을 타겟db 컬럼명으로 변경 위해 소스 컬럼과 대상 컬럼 SWAP
                    reversed_column_mapping.update({src_col: target_col})
                    source_column_name_list = list(
                        reversed_column_mapping.keys())
        source_db_info.update({'columns': source_column_name_list})
        source_df = get_dataframe_from_source_db_info_dict(source_db_info)
        source_df = source_df.rename(columns=reversed_column_mapping)
        return source_df

    def execute_migrate(self, target_db):
        start_time = timeit.default_timer()
        src_db_info = DATABASES[target_db]['MIGRATION_INFO']
        self.migrate_by_mapping_info(target_db, src_db_info)
        end_time = timeit.default_timer()
        print(f'[{self.__class__.__name__}] finished to mgirate, {(end_time-start_time)/60}분 소요')
    
    def migrate_by_mapping_info(target_db, src_db_info):
        pass

    def _insert_data(self, target_db, source_df):
        pass


class DocumentMigrator(BaseMigrator):

    def __init__(self, multiprocess_mode=False, multiprocess_worker_num=int(cpu_count()*0.5)):
        super().__init__(multiprocess_mode, multiprocess_worker_num)

    def migrate_by_mapping_info(self, target_db, migration_info):

        source_df = self._make_source_df_by_migration_info(migration_info)
        # source_df['updated_at'] = datetime.datetime.now()
        # source_df['created_at'] = datetime.datetime.now()
        # 데이터 전처리 필요 시 여기에 함수나 로직 추가
        source_df = default_processing_source_df(source_df)

        print(f'[{self.__class__.__name__}] TRUNCATE document table')
        with connections[target_db].cursor() as cursor:
            cursor.execute("TRUNCATE document")

        print(f'[{self.__class__.__name__}] Migrate {source_df.shape[0]} rows')

        if self.multiprocess_mode:
            print(f'[{self.__class__.__name__}] multiprocess mode on, use {self.multiprocess_worker_num} cpu')
            # from utils.multiprocess_utils import pool_multiprocess
            from multiprocessing import Process
            splitted_source_df = (pd.DataFrame(df) for df in np.array_split(
                source_df, self.multiprocess_worker_num))
            process_list = []
            print(f'[{self.__class__.__name__}] TRUNCATE {target_db}.document Finished')
            for df_ in splitted_source_df:
                p = Process(target=self._insert_data, args=(target_db, df_))
                process_list.append(p)
            for process in process_list:
                process.start()
            for process in process_list:
                process.join()
        else:
            batch_size = 500
            print(f'[{self.__class__.__name__}] bulk insert, batch_size is {batch_size}')
            self._insert_bulk_data(target_db, source_df)
        print(f'[{self.__class__.__name__}] "document table migration Finished"')
    
    def _insert_bulk_data(self, target_db, source_df, batch_size=500):
        insert_target_model = []
        source_list = source_df.to_dict('records')
        for source_dict in source_list:
            m = Document()
            for k,v in source_dict.items():
                m.__setattr__(k,v)
            insert_target_model.append(m)
        Document.objects.using(target_db).bulk_create(insert_target_model, batch_size=batch_size, ignore_conflicts=True)
        print(f'insert {len(source_list)} data done')


    def _insert_data(self, target_db, source_df):

        try:
            access_info = DATABASES[target_db]
            db_engine = create_sqlalchemy_engine(access_info)
            with db_engine.connect() as db_conn:
                source_df.to_sql(DOCUMENT_TABLE_NAME, con=db_conn,
                                 if_exists='append', index=False, chunksize=300, method='multi')
        except IntegrityError:
            with db_engine.connect() as db_conn:
                from documents.models import Document
                Document.objects.using(target_db).filter(
                    pk__in=source_df.doc_id.tolist()).delete()
                source_df.to_sql(DOCUMENT_TABLE_NAME, con=db_conn,
                                 if_exists='append', index=False, chunksize=300, method='multi')


class FrontDataAccessor():
    def __init__(self,target_db='limenet_analysis'):
        self.front_db_info = DATABASES[target_db]['FRONT_DB']
        self.db_engine = create_sqlalchemy_engine(self.front_db_info)

    def get_front_data_df(self,table_name='topic_model',required_cols=['*'],where_condtion=''):
        _table_name = table_name
        if where_condtion is not None and where_condtion != '':
            _table_name = table_name + ' WHERE ' + where_condtion
        source_db_info = {'access_info':self.front_db_info,'table_name': _table_name, 'columns':required_cols}
        front_data_df = get_dataframe_from_source_db_info_dict(source_db_info)
        return front_data_df


class TermMigrator(BaseMigrator):

    def __init__(self, multiprocess_mode=False, multiprocess_worker_num=int(cpu_count()*0.2)):
        super().__init__(multiprocess_mode, multiprocess_worker_num)

    def migrate_by_mapping_info(self, target_db, migration_info):
        source_df = self._make_source_df_by_migration_info(migration_info)
        source_df = default_processing_source_df(source_df)
        source_df['updated_at'] = datetime.datetime.now()
        source_df['created_at'] = datetime.datetime.now()
        if self.multiprocess_mode:
            from utils.multiprocess_utils import pool_multiprocess
            splitted_source_df = (pd.DataFrame(df) for df in np.array_split(
                source_df, self.multiprocess_worker_num))
            insert_data_by_multiprocess = partial(self._insert_data, target_db)
            pool_multiprocess(self.multiprocess_worker_num,
                              insert_data_by_multiprocess, splitted_source_df)
        else:
            self._insert_data(target_db, source_df)
        print('finished')
    
    def _insert_data(self, target_db, source_df):
        access_info = DATABASES[target_db]
        db_engine = create_sqlalchemy_engine(access_info)
        with db_engine.connect() as db_conn:
            source_df.to_sql(TERM_TABLE_NAME, con=db_conn,
                                if_exists='append', index=False, chunksize=10000)

class ElasticManager(BaseMigrator):

    def __init__(self, target_db, multiprocess_mode=False, multiprocess_worker_num=int(cpu_count()*0.2)):
        super().__init__(multiprocess_mode, multiprocess_worker_num)
        self.target_db = target_db
        self.es_info = target_db
        self.alias_name =  self.es_info['INDEX_NAME']
        self._index_name = self.alias_name + '_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        self.es = Elasticsearch(host=self.es_info['HOST'], port=self.es_info['PORT'], http_auth=(
            self.es_info['USER'], self.es_info['PASSWORD']), timeout=5400, max_retries=3, retry_on_timeout=True)

    @property
    def es_info(self):
        return self.__es_info
    
    @es_info.setter
    def es_info(self, target_db):
        if 'ELASTICSEARCH_INFO' in DATABASES[target_db]:
            self.__es_info = DATABASES[target_db]['ELASTICSEARCH_INFO']
    
    def initialize_index_setting(self):
        
        from config.elasticsearch_settings.elasticsearch_index_template import INDEX_TEMPLATE
        template = copy.deepcopy(INDEX_TEMPLATE)
        section_nums = ''
        
        # section 별 doc 빈도 집계가 필요 없을 시 routing 기능 비활성화
        if 'NUMBER_OF_SECTION' in self.es_info:
            section_nums = self.es_info['NUMBER_OF_SECTION']
        if not section_nums or str(section_nums) == '0' or str(section_nums) == '1':
            routing_required = 'false'
        else:
            routing_required = 'true'

        template['mappings']['_routing']['required'] = routing_required

        settings = template['settings']
        settings['number_of_shards'] = section_nums
        analysis_filter = settings['index']['analysis']['filter']
        userdict_path = f'userdict/{self.alias_name}'
        analysis_filter['synonym_filter']['synonyms_path'] = userdict_path+'/synonym.txt'
        analysis_filter['representative_filter']['synonyms_path'] = userdict_path+'/representative.txt'
        analysis_filter['custom_stop_filter']['stopwords_path'] = userdict_path+'/stopword.txt'
        
        tokenizer = settings['index']['analysis']['tokenizer']
        tokenizer['custom_nori_tokenizer']['user_dictionary'] = userdict_path+'/terminology.txt'
        tokenizer['none_mode_nori_tokenizer']['user_dictionary'] = userdict_path+'/terminology.txt'
        # tokenizer['mixed_compound_tokenizer']['user_dictionary'] = userdict_path+'/terminology.txt'
        return template
        
    def create_index(self, index_name=None, delete_if_exists=False):
        if not index_name:
            index_name = self._index_name
        try:
            if delete_if_exists and self.es.indices.exists(index_name):
                print(f'[{self.__class__.__name__}] {index_name} exsits, delete {index_name} and create new index')
                self.es.indices.delete(index_name)
            print(f'[{self.__class__.__name__}] create index as {index_name}')
            res = self.es.indices.create(
                    index=index_name, body=self.initialize_index_setting(), timeout='30m', master_timeout='30m'
                )
            print(f'[{self.__class__.__name__}] create index result : {res}')
            if not res['acknowledged']:
                print(f'[{self.__class__.__name__}] failed to created index in 30minute, please contact engineer : {res}')
                self.es.indices.delete(index=index_name)
            return res
        except RequestError as err:
            print(f'[{self.__class__.__name__}] error occurs: {err}')

    def migrate_by_mapping_info(self, index_name, es_info):
        migration_info = DATABASES[self.target_db]
        target_db_columns = '*'
        if 'target_source_column_mapping' in migration_info['MIGRATION_INFO']:
            target_db_columns = list(migration_info['MIGRATION_INFO']['target_source_column_mapping'].keys())
        
        source_db_info = {'access_info':migration_info, 'table_name': migration_info['MIGRATION_INFO']['table_name'], 'columns': target_db_columns }
        
        source_df = get_dataframe_from_source_db_info_dict(source_db_info)
        
        #  json 텍스트 -> object화 
        source_df = source_df.fillna('')
        if 'prtcp_mp' in source_df.columns:
            source_df['prtcp_mp'] = source_df['prtcp_mp'].map(json.loads)
        if 'ipr' in source_df.columns:            
            source_df['ipr'] = source_df['ipr'].map(json.loads)
        if 'paper' in source_df.columns:
            source_df['paper'] = source_df['paper'].map(json.loads)

        if self.multiprocess_mode:
            splitted_source_df = (pd.DataFrame(df) for df in np.array_split(source_df, self.multiprocess_worker_num))
            from utils.multiprocess_utils import pool_multiprocess
            insert_data_by_multiprocess = partial(insert_elastic_data, index_name,  es_info)
            pool_multiprocess(self.multiprocess_worker_num,
                            insert_data_by_multiprocess, splitted_source_df)
            print('es migration finished')
        else:
            insert_elastic_data(index_name=index_name, es_info=es_info, df=source_df)
    
    def reindexing(self, new_index_name=None, alias_name=None, force_migrate=False, create_dictionary=False):
        """
        현 분석엔진은 1개의 alias 는 1개의 index만 가져야만 작동하는 구조
        함수 호출 시 해당 alias에 속한 모든 index를 모두 지워버리기 때문에 사용에 주의 필요
        force_migrate = True 일 경우, mysql migration도 동시에 이뤄짐
        create_dictionary=True 일 경우, mysql 의 term 관련 데이터를 txt 사전으로 만들어 elasticsearch 호스트에 복사함
        """
        if not new_index_name:
            new_index_name = self._index_name
        if not alias_name:
            alias_name = self.alias_name
        result = {'job':'reindexing es index', 'status': '','detail':{'target_db':alias_name,'index_name':new_index_name}}
        try:
        # 새로운 index 생성
            if create_dictionary:
                from terms.userdict import create_copy_userdict
                target_term_type_list = list(TERM_TYPES.values())
                for _term_type in target_term_type_list:
                    if _term_type == 'compound':
                        print(f'[{self.__class__.__name__}] create {alias_name}, {_term_type} dictionary file is not supported, use terminology as term_type to create userdict')
                        continue
                    print(f'[{self.__class__.__name__}] create {alias_name}, {_term_type} dictionary file and copy to elasticsearh host file system')
                    create_result, copy_result= create_copy_userdict(target_db=alias_name, term_type=_term_type)
                    print(f'[{self.__class__.__name__}] {alias_name}, {_term_type} create and copy dictionary result: creaet={create_result}, copy={copy_result}')
                print(f'[{self.__class__.__name__}] finished to create dictionary file and copy to elasticsearh host file system')
            
            self.create_index(index_name=new_index_name)
            # alias 에서 현재 사용중인 old index_name 가져오기
            cat_client = CatClient(self.es)
            target_alias = cat_client.aliases(name=alias_name, format='json')
            
            # if force_migrate:
            #     print(f'[{self.__class__.__name__}] force_migrate is True, Do migrate document data as migration setting')
            #     docu_migrator = DocumentMigrator(multiprocess_mode=False, multiprocess_worker_num=int(cpu_count()*0.7))
            #     docu_migrator.execute_migrate(self.target_db)
            #     print(f'[{self.__class__.__name__}] document data migration finished')

            # alias에 속한 index가 여러개가 있다면 모두 지우고, mysql db에서 es index로 자료를 migration
            # alias에 속한 index가 없다면,  mysql db에서 es index로 자료를 migration
            if len(target_alias)==1 and not force_migrate:
                # 1개일 경우에는 reindex api로 indexing 진행
                old_index = target_alias[0]['index']
                self.es.reindex(
                    body={
                        'source': {
                            'index': f'{old_index}'
                        },
                        'dest': {
                            'index': f'{new_index_name}'
                        }
                    },
                    timeout='1h'
                )
            else:
                print('[force migrate] delete all alias related index and insert new data from db')
                self.migrate_by_mapping_info(index_name=new_index_name, es_info=self.es_info)
                
            # old index 의 alias 삭제 new index 에 alias 생성
            alias_actions = {'actions':[]}
            alias_actions['actions'].append(
                {'add': {'index': new_index_name, 'alias': alias_name}}
            )
            old_index_list = []
            for alias in target_alias:
                old_index = alias['index']
                remove_alias = {'remove': {'index':old_index,'alias': alias_name}}
                alias_actions['actions'].append(remove_alias)
                old_index_list.append(old_index)
            res = self.es.indices.update_aliases(alias_actions)
            
            if res['acknowledged']:
                # 기존 인덱스 제거
                for old_index in old_index_list:
                    delete_res = self.es.indices.delete(index=old_index)
                    print(f'{old_index} delete result: {delete_res}')
                # 오류로 인해 남아있는 기존 인덱스도 제거
                for _index_name, _index_with_alias in self.es.indices.get_alias(f'{alias_name}*').items():
                    if _index_with_alias['aliases']:
                        if new_index_name != _index_name:
                            delete_res = self.es.indices.delete(index=_index_name)
                            print(f'{_index_name} delete result: {delete_res}')
            print('reindexing finished')
            # 캐쉬 데이터 삭제
            self.es.delete_by_query(index='analysis_cache',body={'query':{'match_all':{}}},ignore=[400,404])
            result['status'] = 'success'
            return result
        except Exception as e:
            print(e)
            if self.es.indices.exists(new_index_name):
                print(f'[{self.__class__.__name__}] error occurs, delete {new_index_name} ')
                self.es.indices.delete(new_index_name)
            result['status'] = 'failed'
            result['detail'].update({'error':e})
            return result

    def auto_initialize_index(self): 
        initialize_result = None
        res = self.create_index()
        if not res['acknowledged']:
            print('failed to initialize')
            return initialize_result
        self.es.indices.put_alias(index=self._index_name, name=self.alias_name)
        self.migrate_by_mapping_info(index_name=self.alias_name, es_info=self.es_info)
        
    def refresh_index(self):
        result = self.es.indices.close(index=self._index_name)
        if result['acknowledged']:
            open_result = self.es.indices.open(index=self._index_name)
            if open_result['acknowledged']:
                print('refreshed')


class ElasticQuery(ElasticManager):
    def __init__(self, target_db, multiprocess_mode=False, multiprocess_worker_num=1):
        super().__init__(target_db, multiprocess_mode, multiprocess_worker_num)
        from config.elasticsearch_settings.cache_index_template import CACHE_INDEX_TEMPLATE
        self.cache_index_template = CACHE_INDEX_TEMPLATE
    
    def get_all_docs(self, source=[""]):
        res = helpers.scan(self.es, index=self.alias_name, query={"_source": source, "query": {"match_all":{}}}, scroll="15m")
        return res 

    def get_all_docs_by_search_text(self, search_text, source=[""],fields=["title","content"]):
        res = helpers.scan(self.es, index=self.alias_name, query={"_source": source, "query": {"multi_match":{"query":search_text,"fields":fields}}}, scroll="15m")
        return res 

    def get_all_docs_by_query_string(self, search_text, source=[""],fields=["title^2","content"]):
        res = helpers.scan(self.es, index=self.alias_name, 
        query={
            "_source": source,
            "query": {
                "bool": {
                "must": [
                    {
                    "query_string": {
                        "fields": fields,
                        "query": f"{search_text}",
                        "auto_generate_synonyms_phrase_query": False
                        }
                    } 
                ]
                }  
            }
        }, 
        scroll="15m")
        return res 

    def get_docs_by_search_text(self, search_text, size=1000, source=[""], fields=["title^2", "content"]):
        result = self.es.search(
            index = self.alias_name,
            size = size,
            body = {
                "_source": source,
                "query": {
                    "multi_match": {
                        "query": search_text,
                        "fields": fields
                    }
                }
            }
        )
        return result["hits"]["hits"]
    
    def get_docs_by_keyword_field(self, target_index, keyword_field, keyword_list, size=500, source=[""]):
        search_format = {
            "_source":source,
            "query": {
                "bool":{
                    "should": [
                    ]
                }
            }
        } 
        for idx in range(0,len(keyword_list),60000):
            search_format["query"]["bool"]["should"].append({"terms":{f"{keyword_field}.keyword": keyword_list[idx:idx+60000]}})
        result = self.es.search(
            index = target_index,
            size = size,
            body = search_format
        )
        return result["hits"]["hits"]
    
    def validate_query_string(self, query_string):
        query_string_format = {
            "query": {
                "bool": {
                "must": [
                    {
                    "query_string": {
                        "fields": ["kor_kywd^1.5","eng_kywd^1.5","kor_pjt_nm","rsch_goal_abstract","rsch_abstract","exp_efct_abstract"],
                        "query": f"{query_string}",
                        "auto_generate_synonyms_phrase_query": False
                        }
                    } 
                ]
                }  
            }
        }
        valid = True
        error_detail = ''
        valid_result = self.es.indices.validate_query(index='kistep_sbjt', body=query_string_format, explain=True)
        if not valid_result['valid']:
            valid = valid_result['valid']
            error_detail = valid_result['explanations'][0]['error']
        return {'valid':valid, 'error': error_detail}

    def get_docs_by_doc_ids(self, doc_ids, source=['*']):
        combined_res_format = {
            "took": 0,
            "timed_out": False,
            "hits": {"total":{"value":0, "relation":"eq"},
                    "max_score": 1,
                    "hits": []
                    }
        }
        timed_out_check = False
        for idx in range(0, len(doc_ids),10000):
            query_format = {
                "_source": source,
                "size": 0,
                "query": {
                        "terms": {
                            "doc_id": []
                        }
                    }
            }
            sub_docs = doc_ids[idx:idx+10000]
            query_format["query"]["terms"]["doc_id"].extend(sub_docs)
            query_format["size"] = len(sub_docs)
            sub_res = self.es.search(index=self.alias_name,body=query_format)
            combined_res_format["took"] += sub_res["took"]
            combined_res_format["hits"]["total"]["value"] += sub_res["hits"]["total"]["value"]
            combined_res_format["hits"]["hits"].extend(sub_res["hits"]["hits"])
            if sub_res["timed_out"]:
                timed_out_check = True
        combined_res_format.update({"timed_out":timed_out_check})    
        return combined_res_format
    
    def get_percentile_score(self, query, filter_percent):
        res = self.es.search(index=self.alias_name, body={
            "size": 0,
            "_source": [""],
            "query": query,
            "aggs":{
                "load_by_ranks":{
                    "percentiles": {
                        "script": "_score",
                        "percents": [100-filter_percent]
                    }
                }
            }
        })
        min_score = list(res['aggregations']['load_by_ranks']['values'].values())[0]
        try:
            if min_score is None:
                return 0
            min_score = int(min_score)
            return min_score
        except ValueError:
            return 0
        except TypeError:
            return 0 

    def scroll_docs_by_query(self, query, source=['*'], min_score=0):
        if not min_score:
            min_score = 0
        combined_res_format = {
            "took": 0,
            "timed_out": False,
            "hits": {"total":{"value":0, "relation":"eq"},
                    "max_score": 1,
                    "hits": []
                    }
        }            
        res = self.es.search(index=self.alias_name, body={
                "size": 10000,
                "_source": source,
                "min_score": min_score,
                "query": query,
            },
            scroll='10s'
        )
        old_scroll_id = res["_scroll_id"]
        combined_res_format["took"] += res["took"]
        combined_res_format["hits"]["total"]["value"] += res["hits"]["total"]["value"]
        combined_res_format["hits"]["hits"].extend(res["hits"]["hits"])
        
        while len(res["hits"]["hits"]):
            res = self.es.scroll(
                scroll_id=old_scroll_id,
                scroll="10s"
            )
            old_scroll_id = res["_scroll_id"]
            combined_res_format["took"] += res["took"]
            combined_res_format["hits"]["total"]["value"] += res["hits"]["total"]["value"]
            combined_res_format["hits"]["hits"].extend(res["hits"]["hits"])
        return combined_res_format

    def get_docs_by_full_query(self, query, doc_size=300):
        return self.es.search(index=self.alias_name, body=query, size=doc_size)        
        
    def get_docs_by_query_string(self, target_db, query_string, source=['*'], doc_size=1000):
        query_string_format = {
            "_source" : source,
            "query": {
                "bool": {
                    "must": [
                        {
                        "query_string": {
                            "fields": ["kwd^1.5","han_sbjt_nm","rsch_gole_cn","rsch_rang_cn","expe_efct_cn"],
                            "query": f"{query_string}",
                            "auto_generate_synonyms_phrase_query": False
                            }
                        } 
                    ]
                }  
            }
        }
        res = self.es.search(index=target_db,body=query_string_format, size=doc_size)
        return res

    def get_tokens_from_doc_info(self, doc_info, fields=["analysis_target_text"], field_statistics="false", term_statistics="true", positions="true", offsets="false"):
        """
        request: doc_info as below
        doc_info: 
        [
            {   
                "_index":""
                "_id":"",
                "routing": ""
            },
            {
                ...
            }
        ]
        """
        target_doc_info = []
        if not doc_info or len(doc_info)==0:
            return []
        for doc in doc_info:    
            mterm_info = {
                "_index":doc["_index"], 
                "_id": doc["_id"],
                "fields": fields,                 
                "term_statistics": term_statistics,
                "field_statistics": field_statistics,
                "positions": positions,
                "offsets": offsets,
                "routing": doc["_routing"]
            }
            target_doc_info.append(mterm_info)

        result = self.es.mtermvectors(
            index=self.alias_name,
            body={
                "docs": target_doc_info
            }
        )
        return result["docs"]

    def trim_text_under_max_clauses_size(self, search_text, field):
        analyze_result = self.es.indices.analyze(
            index=self.alias_name, body=dict(field=field, text=search_text)
        )
        if len(analyze_result['tokens']) > MAX_CLAUSE_COUNT:
            over_tokens = analyze_result['tokens'][MAX_CLAUSE_COUNT:]
            if len(over_tokens)>0:
                target_offset = over_tokens[0]['start_offset']
                return search_text[:target_offset]
        return search_text

    def tokenize_text(self, search_text, field="title"):
        analyze_result = self.es.indices.analyze(
            index=self.alias_name, body=dict(field=field, text=search_text)
        )
        tokens = [tk["token"] for tk in analyze_result["tokens"]]
        if len(tokens)>1:
            tokens = sorted(list(set(tokens)))
        return tokens

    def unpacking_termsvector(self, term_vector_result):
        unpack_data = []
        for doc in term_vector_result:
            _index = doc["_index"]
            doc_id = doc["_id"]
            term_vectors = doc["term_vectors"]
            if term_vectors and len(term_vectors)==0:
                continue
            for field_name, field in term_vectors.items():
                for term, freq_dict in field["terms"].items():
                    row = {}
                    row.update({"index": _index, "doc_id": doc_id, "term": term, "field":field_name})
                    for freq_type, freq in freq_dict.items():
                        if freq_type == "tokens":
                            position_list =[]
                            token_info_list = []
                            for token in freq:
                                for token_type, token_value in token.items():
                                    if token_type == "position":
                                        position_list.append(token_value)
                                    else:
                                        token_info_list.append({token_type:token_value})
                            row.update({"position": position_list, "token_info": token_info_list})
                        else:
                            row.update({freq_type:freq})

                    unpack_data.append(row)
        return unpack_data

    def calculate_subclass_doc_freq(self, search_target_list, target_category='subclass'):
        """ 
        특정 단어가 문서의 제목이나 본문에 등장하며, 특정 카테고리 타입인 문서의 전체 개수

        Parameters
        ----------
        search_target_list: list[dict], index-term-'categroy_column_name' 키를 갖는 dictionary 로 이루어진 list
            ex) [{'index':'docu_patent','term':'인공지능','subclass':'H01L'},..]
        
        Returns
        -------
        search_target_list: list[dict], parameter의 list 내 dictionary 속에 {'cdfreq':문서개수} 데이터 추가 후 반환
        *cdfreq => categry document frequency
        """
        msearch_format = []
        for search_condition in search_target_list:
            _index = {"index":search_condition["index"]}
            _query =  {"query": {"bool": {"filter": [{"term": {"doc_subclass": search_condition[target_category]}},{"bool": {"should":[{"term":{"title":search_condition["term"]}},{"term": {"content":search_condition["term"]}}]}}]}},
                "_source":[""],
                "size":0,
                "aggs":{"cdfreq":{"value_count":{"field":"doc_id"}}}}
            msearch_format.append(_index)
            msearch_format.append(_query)
        res = self.es.msearch(body=msearch_format)
        doc_freq_list = []
        for response in res["responses"]:
            doc_freq = response['aggregations']['cdfreq']['value']
            doc_freq_list.append(doc_freq)
        for i in range(len(search_target_list)):
            search_target_list[i].update({'cdfreq':doc_freq_list[i]})
        return search_target_list

    def calculate_yearly_category_doc_freq(self, search_target_list, target_category='doc_subclass'):
        """ 
        특정 단어가 문서의 제목이나 본문에 등장하는 문서의 연도별, 카테고리별 개수

        Parameters
        ----------
        search_target_list: list[dict], index-term 키를 갖는 dictionary 로 이루어진 list
            ex) [{'index':'docu_patent','term':'인공지능'},..]
        
        Returns
        -------
        doc_freq_list: list[dict], [{'term':검색단어,'pyear':연도, target_category:카테고리명, 'doc_freq':문서개수}]
            ex) [{'term': '인공지능', 'pyear': '1987', 'doc_subclass': 'A61B', 'doc_freq': 1},...]
        """
        msearch_format = []
        for search_condition in search_target_list:
            _index = {"index":search_condition["index"]}
            _query =  {"query": {"bool": {"should":[{"term":{"analysis_target_text":search_condition["term"]}}]}},
                "_source":[""],
                "size":0,
                "aggs":{"date_year":{"date_histogram":{"field":"stan_yr","calendar_interval":"year"},"aggs":{"count_doc_category":{"terms":{"field":target_category,"size":10000}}}}}}
            msearch_format.append(_index)
            msearch_format.append(_query)
        res = self.es.msearch(body=msearch_format)
        doc_freq_list = []
        error_list = []
        for res_idx, response in enumerate(res["responses"]):
            if 'error' in response:
                error_list.append(response)
                continue
            buckets = response["aggregations"]["date_year"]["buckets"]
            for each_bucket in buckets:
                _year = each_bucket["key_as_string"]
                _dfreq = each_bucket["doc_count"]
                _cdfreq_bucket = each_bucket["count_doc_category"]["buckets"]
                for _cdfreq in _cdfreq_bucket:
                    sub_res = {}
                    _category = _cdfreq["key"]
                    _cdfreq = _cdfreq["doc_count"]
                    sub_res = {"term":search_target_list[res_idx]["term"], "pyear":_year, 'category':_category, "doc_freq":_cdfreq}
                    doc_freq_list.append(sub_res)
        return doc_freq_list


    def get_analysis_cache_by_search_keys(self, target_db, search_keys, analysis_type, doc_size, source=['*']):
        try:
            if not isinstance(search_keys,str):
                search_keys = json.dumps(search_keys,ensure_ascii=False)
            result = self.es.search(
                index="analysis_cache",
                size=1,
                body={
                    "_source": source,
                    "sort": [
                        {
                        "timestamp": {
                            "order": "desc"
                        }
                        }
                    ],
                    "query": {
                        "bool": {
                            "must": [
                                {"term": {"target_db": target_db}},
                                {"term": {"search_keys": search_keys}},
                                {"term": {"analysis_type": analysis_type}},
                                {"term": {"doc_size": doc_size}},
                            ]
                        }
                    },
                },
            )
            if result['hits']['total']['value'] == 0:
                return None
            return result['hits']['hits'][0]
        except elasticsearch.NotFoundError as e:
            return None
        except json.decoder.JSONDecodeError:
            return None
    
    def insert_analysis_cache(self, target_db, search_keys, analysis_type, doc_size, analysis_result):
        try:
            if not self.es.indices.exists(index='analysis_cache'):
                self.es.indices.create(
                    index='analysis_cache', body=self.cache_index_template, timeout='3m', master_timeout='3m', ignore=[400,404]
                )
            res = 'failed'
            body = {}
            if not isinstance(search_keys,str):
                search_keys = json.dumps(search_keys,ensure_ascii=False)
            if not search_keys :
                return res
            body.update({
                            'target_db':target_db, 'search_keys':search_keys, 
                            'analysis_type': analysis_type, 'doc_size': doc_size, 'analysis_result':analysis_result,
                            'timestamp': return_timestamp()
                        })
            exists_cache = self.get_analysis_cache_by_search_keys(target_db=target_db, search_keys=search_keys, analysis_type=analysis_type, doc_size=doc_size)
            if exists_cache:
                self.es.delete(index='analysis_cache',id=exists_cache['_id'],ignore=[400,404])
            res = self.es.index(index="analysis_cache", body=body)
            return res
        except json.decoder.JSONDecodeError:
            raise RequestError('query malformed')
        except TransportError as te:
            raise RequestError(te)


    def delete_index(self, index_name='analysis_cache'):
        self.es.indices.delete(index_name, ignore=[400,404])

    def delete_all_data(self, index_name):
        self.es.delete_by_query(
            index=index_name,
            body={
                "query": {
                    "match_all": {}
                }
            }
        )

    def delete_data_by_field_terms(self, index_name, field_name, terms):
        self.es.delete_by_query(
            index = index_name,
            body = {
                "query":{
                    "terms":{
                        field_name: terms
                    }
                }
            },
            ignore=[400,404]
        )

if __name__ == '__main__':
    import sys
    import os
    from pathlib import Path
    APP_DIR = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(APP_DIR))
    os.environ['DJANGO_SETTINGS_MODULE'] = 'config.settings.base'
    os.environ['DJANGO_ALLOW_ASYNC_UNSAFE'] = 'true'
    import django
    django.setup()
    from config.constants import DATABASES, DOCUMENT_TABLE_NAME,TERM_TABLE_NAME, TERM_TYPES, DOCUMENT_DB_NAMES
    from documents.models import Document
    from django.db import connections

    import timeit
    start_time = timeit.default_timer()

    # for target_db in ['docu_book']:
    #     start_time = timeit.default_timer()
    #     em = ElasticManager(target_db=target_db)
    #     em.reindexing(force_migrate=True, create_dictionary=True)
    #     print(timeit.default_timer()-start_time)

    # =====================
    # for target_db in ['docu_test','docu_patent','docu_thesis','docu_subject']:
    #     docu_mig = DocumentMigrator(multiprocess_mode=True, multiprocess_worker_num=30)
    #     docu_mig.execute_migrate(target_db=target_db)

    # term_mig = TermMigrator(multiprocess_mode=True)
    # term_mig.execute_migrate(target_db='term_default')

    # em.migrate_by_mapping_info(index_name='docu_test',es_info=em.es_info)
    # em.migrate_by_mapping_info()
    # docu_migrator = DocumentMigrator(multiprocess_mode=True,
    #                          multiprocess_worker_num=40)
    # docu_migrator.execute_migrate(target_db)
    # term_migrator = TermMigrator(multiprocess_mode=True,
    #                          multiprocess_worker_num=40)
    # term_migrator.execute_migrate(target_db)
    
    # player temp insert
    # sbjt_player_df = pd.read_parquet('/data/temp_parquet/sbjt_player_tmp.parquet')
    # columns_name = {
    #     'sbjt_id': 'doc_id',
    #     'sbjt_name': 'title',
    # }
    # sbjt_player_df = sbjt_player_df.rename(columns_name,axis=1)
    # es_manager = ElasticManager(target_db='docu_subject')
    
    # insert_elastic_data(index_name='player_info', df=sbjt_player_df, es_info=es_manager.es_info, max_retries=5)
    target_db = 'kistep_sbjt'
    # docu_migrator = DocumentMigrator(multiprocess_mode=False, multiprocess_worker_num=int(cpu_count()*0.7))
    # docu_migrator.execute_migrate(target_db=target_db)
    # es_manager = ElasticManager(target_db=target_db)
    # es_manager.reindexing(force_migrate=True, create_dictionary=True)
    fda = FrontDataAccessor(target_db=target_db)
    fda.get_front_data_df(table_name='topic_model', required_cols=['topic_id','topic_model','last_modf_dt'], where_condtion='last_modf_dt>2021-12-03')
    print(timeit.default_timer()-start_time, "min:", (timeit.default_timer()-start_time)/60)