# from django.db.models.expressions import F
# import pandas as pd
# from django.core.cache import cache

# if __name__ !='__main__':
#     from db_manager.managers import ElasticQuery
#     from terms.models import EdgelistTerm
#     from config.constants import WORDCOUNT_EDGELIST_COLUMN_MAPPING, SNA_EDGELIST_COLUMN_MAPPING, SUBJECT_PLAYER_EDGELIST_COLUMN_MAPPING

# class BaseEdgelist():
#     """
#     문서-용어-빈도의 정보를 담고 있는 dataframe 생성 클래스
#     처리 순서: 검색문장 -> 문서 검색 -> 문서 내 단어 elasticsearch 조회 -> 대표어,동의어,불용어 처리 -> 기술용어 외 단어 제거 -> 분석 모듈 요구 사항에 맞게 후처리
#     """

#     def _make_edgelist_from_es_docs_format(self, target_db, docs, offsets=False, positions=True):
#         """
#         elasticsearch 검색 결과에서 dataframe 으로 변환하는 공통 사용 함수 
        
#         Parameters
#         ----------
#         target_db : str, 분석 대상 document index ex) docu_patent, docu_thesis, etc
#         docs : list, elasticsearch search api 반환 결과 형태
#             sample: [{'_index': 'docu_test', '_type': '_doc', '_id': '1019790004128', '_score': None, '_routing': 'A', '_source': {...}, 'sort': [...]},{..}]
#         size : int, default = 500, 검색 문서 개수 (검색 score 상위 n 개)
#         offsets : bool, default = False , 문서 내 offset(단어 시작 위치, 종료 위치) 반환
#         positions : bool, default = True, 문서 내 position 정보 반환
        
#         Returns
#         -------
#         edgelist_df : pandas.DataFrame
#             df columns : 
#                 doc_id: str, 문서번호
#                 index: str, elasticsearch index 이름
#                 term: str, 단어
#                 field: str , 단어가 포함된 elasticsearch index mapping 필드
#                 doc_freq: int, shard 내 해당 단어 포함 문서 갯수
#                 ttf: int, shard 내 해당 단어의 전체 갯수
#                 term_freq: int, 문서 내 해당 단어의 갯수
#                 position: 문서 내 위치
#                 token_info: list(dict), 문서 내 offset 정보
#                 section: str, routing 정보
#                 subclass: str, doc_subclass 정보
#                 score: float, 검색 시 단어의 score(bm25 기준)
#         """
#         es_query = ElasticQuery(target_db)
#         docs_df = pd.DataFrame(docs)
#         if docs_df.empty:
#             return pd.DataFrame()
#         docs_df['doc_subclass'] = docs_df['_source'].map(lambda x: x['doc_subclass'])
#         docs_df['publication_date'] = docs_df['_source'].map(lambda x: x['publication_date'])
#         docs_df['title'] = docs_df['_source'].map(lambda x: x['title'])
#         # 현재는 _id = doc_id라서 불필요한 연산을 없애기 위해 _id 값을 index값으로 사용.
#         # 추후 doc_id != _id 일 경우 해당 부분 반영해줘야함
#         # docs_df['doc_id'] = docs_df['_source'].map(lambda x: x['doc_id'])
#         docs_df = docs_df.set_index('_id')
#         term_vectors = es_query.get_tokens_from_doc_info(docs, offsets=offsets, positions=positions)
#         unpacked_term_vectors = es_query.unpacking_termsvector(term_vectors)
#         edgelist_df = pd.DataFrame(unpacked_term_vectors)
#         edgelist_df = edgelist_df.set_index(['doc_id'])
#         edgelist_df['section'] = docs_df['_routing']
#         edgelist_df['subclass'] = docs_df['doc_subclass']
#         edgelist_df['score'] = docs_df['_score']
#         edgelist_df['title'] = docs_df['title']
#         edgelist_df['publication_date'] = docs_df['publication_date']
#         edgelist_df = edgelist_df.reset_index()
#         return edgelist_df

#     def get_ego_edgelist(self, target_db, search_text, is_query_string=False, size=500, offsets=False, positions=True):
#         """
#         검색 문장에 해당하는 edgelist 생성

#         Parameters
#         ----------
#         target_db : str, 분석 대상 document index ex) docu_patent, docu_thesis, etc
#         search_text : str, 검색 문장, 단어 등 텍스트
#         size : int, default = 500, 검색 문서 개수 (검색 score 상위 n 개)
#         offsets : bool, default = False , 문서 내 offset(단어 시작 위치, 종료 위치) 반환
#         positions : bool, default = True, 문서 내 position 정보 반환
        

#         Returns
#         -------
#         edgelist_df : pandas.DataFrame
#             _make_edgelist_from_es_docs_format 함수 return 값과 동일
#         """
#         es_query = ElasticQuery(target_db)
#         source = ['doc_id','doc_subclass','title','publication_date']
#         # query string 으로 edgelist 생성
#         if is_query_string:
#             search_result = es_query.get_docs_by_query_string(target_db=target_db, query_string=search_text, source=source, doc_size=size)
#             docs = search_result['hits']['hits']
#         else:
#             docs = es_query.get_docs_by_search_text(search_text=search_text, size=size, source=source)
#         edgelist_df = self._make_edgelist_from_es_docs_format(target_db, docs, offsets, positions)
#         return edgelist_df

#     def get_refined_ego_edgelist(self, target_db, search_text, is_query_string=False, size=500, offsets=False, positions=True, return_original=False):
#         """
#         Parameters
#         ----------
#         get_ego_edgelist 함수와 동일

#         Returns
#         -------
#         edgelist_df : pandas.DataFrame
#             _make_edgelist_from_es_docs_format 함수 return 값과 동일
#         """
#         edgelist_df = self.get_ego_edgelist(target_db, search_text, is_query_string=is_query_string, size=size, offsets=offsets, positions=positions)
#         if edgelist_df.empty:
#             if return_original:
#                 return edgelist_df, edgelist_df
#             return edgelist_df
#         # edgelist 용어 리스트 불러오기
#         # target db edgelist 용어 (현재는 subclass나 section 별로 용어를 구분하지 않고 있음)
#         # start_time = timeit.default_timer()
#         target_db_term = cache.get(f'{target_db}_edgelist_term')
#         if not target_db_term:
#             target_db_term = set(EdgelistTerm.objects.using(target_db).all().values_list('term',flat=True))
#             cache.set(f'{target_db}_edgelist_term',target_db_term)
        
#         # start_time = timeit.default_timer()
#         ego_edgelist_df = edgelist_df[edgelist_df['term'].isin(target_db_term)]
#         # print(f'{target_db}, isin',timeit.default_timer()-start_time)
#         # 동의어/불용어/대표어 처리 or 엘라스틱서치 인덱싱 시 적용 추가 예정
#         if return_original:
#             return edgelist_df, ego_edgelist_df
#         return ego_edgelist_df

#     def get_all_edgelist(self, target_db, chunk_size=10000, offsets=False, positions=True):
#         """
#         target_db와 연결되어 있는 elasticsearch index 의 모든 문서의 단어/빈도수

#         Parameters
#         ----------
#         target_db : str, 분석 대상 document index ex) docu_patent, docu_thesis, etc
#         chunk_size : int, default = 10000, elasticsearch 에서 문서 내 단어를 반환 받을 때 한번의 request에 담을 문서 개수. 부하 방지위해 사용 
#         offsets : bool, default = False , 문서 내 offset(단어 시작 위치, 종료 위치) 반환
#         positions : bool, default = True, 문서 내 position 정보 반환
        
#         Returns
#         -------
#         edgelist_df : pandas.DataFrame
#             _make_edgelist_from_es_docs_format 함수 return 값과 동일
#         """
#         es_query = ElasticQuery(target_db)
#         all_docs_generator = es_query.get_all_docs(source=['doc_id','doc_subclass','title','publication_date'])
#         loop_index = True
#         term_vector_df_list = []
#         while loop_index:
#             docs_gen_list = []
#             try:
#                 for idx in range(chunk_size):
#                     docs_gen_list.append(next(all_docs_generator))
#                 edgelist_df = self._make_edgelist_from_es_docs_format(target_db, docs_gen_list, offsets, positions)
#                 term_vector_df_list.append(edgelist_df)
#             except StopIteration:
#                 edgelist_df = self._make_edgelist_from_es_docs_format(target_db, docs_gen_list, offsets, positions)
#                 term_vector_df_list.append(edgelist_df)
#                 loop_index = False
#         combiend_term_vector_df = pd.DataFrame()
#         if len(term_vector_df_list)>0:
#             combiend_term_vector_df = pd.concat(term_vector_df_list, join='outer', ignore_index=True, axis=0)
#             combiend_term_vector_df['score'] = combiend_term_vector_df['score'].fillna(0)
#         return combiend_term_vector_df


# class SnaEdgelist(BaseEdgelist):
#     """
#     WordExtractor 분석 모듈에서 사용할 수 있는 형태로 edgelist 변환하는 클래스.
#     EldgelistManager 로직 그대로 사용하고, 추가로 컬럼명 변환, title field의 position 정보 제거함
#     """
#     def __init__(self):
#         super().__init__()
    
#     def _transform(self, df):
#         if df.empty:
#             return df
#         if 'field' in df.columns:
#             title_position = df.loc[df['field'] == 'title']['position'].map(lambda x: [])
#             df.loc[title_position.index]['position'] = title_position
#         df = df.rename(SNA_EDGELIST_COLUMN_MAPPING, axis=1).drop('DELETE',axis=1)
#         return df

#     def get_all_edgelist(self, target_db, chunk_size=10000, offsets=False, positions=True):
#         edgelist_df = super().get_all_edgelist(target_db, chunk_size=chunk_size, offsets=offsets, positions=positions)
#         edgelist_df = self._transform(edgelist_df)
#         return edgelist_df

#     def get_sna_ego_edgelist(self, target_db, search_text, is_query_string=False, size=500, offsets=False, positions=True):
#         ego_edgelist_df = self.get_ego_edgelist(target_db=target_db, search_text=search_text, is_query_string=is_query_string,size=size, offsets=offsets, positions=positions)
#         ego_edgelist_df = self._transform(ego_edgelist_df)
#         return ego_edgelist_df

#     def get_sna_refined_ego_edgelist(self, target_db, search_text, is_query_string=False, size=500, offsets=False, positions=True, return_original=True):
#         ego_edgelist_df, refined_ego_edgelist = self.get_refined_ego_edgelist(target_db=target_db, search_text=search_text, is_query_string=is_query_string, size=size, return_original=return_original)
#         ego_edgelist_df = self._transform(ego_edgelist_df)
#         refined_ego_edgelist = self._transform(refined_ego_edgelist)
#         return ego_edgelist_df, refined_ego_edgelist

# class PlayerEdgelist(BaseEdgelist):
#     """
#     Player 네트워크 분석 모듈에서 사용할 수 있는 형태로 edgelist 변환하는 클래스.
#     EldgelistManager 로직 그대로 사용하고, 추가로 Player 정보를 추가함
#     """
#     def __init__(self):
#         super().__init__()

#     def _make_player_df_from_es_docs_format(self, docs, source_fields):
#         """
#         elasticsearch player 검색 결과에서 dataframe 으로 변환
        
#         Parameters
#         ----------
#         docs : list, elasticsearch search api 반환 결과 형태
#             sample: [{'_index': 'docu_test', '_type': '_doc', '_id': '1019790004128', '_score': None, '_routing': 'A', '_source': {...}},{..}]
        
#         Returns
#         -------
#         edgelist_df : pandas.DataFrame
#             df columns : 
#                 given source_fields parameters
#         """
#         docs_df = pd.DataFrame(docs)
#         if docs_df.empty:
#             return pd.DataFrame()
#         player_df = pd.DataFrame()
#         for source_field in source_fields:
#             player_df[source_field] = docs_df['_source'].map(lambda x: x[source_field])
#         return player_df

#     def _transform(self, df):
#         if df.empty:
#             return df
#         if 'field' in df.columns:
#             title_position = df.loc[df['field'] == 'title']['position'].map(lambda x: [])
#             df.loc[title_position.index]['position'] = title_position
#         df = df.rename(SUBJECT_PLAYER_EDGELIST_COLUMN_MAPPING, axis=1).drop('DELETE',axis=1)
#         return df

#     def get_custom_refined_ego_edgelist(self, target_db, search_text, is_query_string=False, size=500, offsets=False, positions=True, return_original=True):
#         ego_edgelist_df, refined_ego_edgelist = self.get_refined_ego_edgelist(target_db=target_db, search_text=search_text, is_query_string=is_query_string, size=size, return_original=return_original)
#         source_fields = ['doc_id','sbjt_supervisor_orgn_id','sbjt_supervisor_orgn_name','sbjt_executor_orgn_id','sbjt_executor_orgn_name']
#         if ego_edgelist_df.empty or refined_ego_edgelist.empty:
#             return ego_edgelist_df, refined_ego_edgelist
#         es_query = ElasticQuery(target_db)
#         player_info = es_query.get_docs_by_keyword_field(
#                 target_index = 'player_info',keyword_field='doc_id', keyword_list=ego_edgelist_df.doc_id, size=size,
#                 source = source_fields
#             )
#         player_df = self._make_player_df_from_es_docs_format(docs=player_info, source_fields=source_fields)
#         if player_df.empty:
#             return player_df, player_df
#         ego_edgelist_df = ego_edgelist_df.set_index('doc_id')
#         refined_ego_edgelist = refined_ego_edgelist.set_index('doc_id')
#         player_df = player_df.set_index('doc_id')
#         ego_edgelist_df[player_df.columns] = player_df[player_df.columns]
#         refined_ego_edgelist[player_df.columns] = player_df[player_df.columns]
#         ego_edgelist_df = ego_edgelist_df.reset_index()
#         ego_edgelist_df = ego_edgelist_df.dropna()
#         refined_ego_edgelist = refined_ego_edgelist.reset_index()
#         refined_ego_edgelist = refined_ego_edgelist.dropna()
#         ego_edgelist_df = self._transform(ego_edgelist_df)
#         refined_ego_edgelist = self._transform(refined_ego_edgelist)
#         return ego_edgelist_df, refined_ego_edgelist    

# class WordCountEdgelist(BaseEdgelist):
#     """
#     """
#     def __init__(self):
#         super().__init__()
    
#     def _transform(self, df):
#         df = df.rename(WORDCOUNT_EDGELIST_COLUMN_MAPPING, axis=1).drop('DELETE',axis=1)
#         return df

#     def get_term_doc_id_mapping_from_edgelist(self, df):
#         try:
#             df = df[['term','doc_id']]
#             df = df.drop_duplicates()
#             return df.groupby('term')['doc_id'].apply(list).to_dict()
#         except KeyError:
#             return {}

#     def get_wc_ego_edgelist(self, target_db, search_text, is_query_string=False, size=500, target_category='doc_subclass', offsets=False, positions=False):
#         es_query = ElasticQuery(target_db)

#         ego_edgelist_df = self.get_ego_edgelist(target_db=target_db, search_text=search_text, is_query_string=is_query_string, size=size, offsets=offsets, positions=positions)
#         term_doc_id_mapping = self.get_term_doc_id_mapping_from_edgelist(ego_edgelist_df)
#         if ego_edgelist_df.empty:
#             return ego_edgelist_df, term_doc_id_mapping
#         ego_edgelist_df = self._transform(ego_edgelist_df)
#         ego_edgelist_df = ego_edgelist_df.drop_duplicates()
#         yearly_category_doc_freq_list = es_query.calculate_yearly_category_doc_freq(ego_edgelist_df.to_dict('records'), target_category=target_category)
#         ego_edgelist_df = pd.DataFrame(yearly_category_doc_freq_list)
#         return ego_edgelist_df, term_doc_id_mapping

#     def get_wc_refined_ego_edgelist(self, target_db, search_text, is_query_string=False, size=500, target_category='doc_subclass', offsets=False, positions=False):
#         es_query = ElasticQuery(target_db)
        
#         refined_ego_edgelist = self.get_refined_ego_edgelist(target_db=target_db, search_text=search_text, is_query_string=is_query_string, size=size, offsets=offsets, positions=positions)
#         term_doc_id_mapping = self.get_term_doc_id_mapping_from_edgelist(refined_ego_edgelist)

#         if refined_ego_edgelist.empty:
#             return refined_ego_edgelist,  term_doc_id_mapping
#         refined_ego_edgelist = self._transform(refined_ego_edgelist)
#         refined_ego_edgelist = refined_ego_edgelist.drop_duplicates()
#         yearly_category_doc_freq_list = es_query.calculate_yearly_category_doc_freq(refined_ego_edgelist.to_dict('records'), target_category=target_category)
#         refined_ego_edgelist = pd.DataFrame(yearly_category_doc_freq_list)
#         return refined_ego_edgelist, term_doc_id_mapping

# class LimitedWordCountEdgelist(BaseEdgelist):
#     def __init__(self):
#         super().__init__()
    
#     def get_limited_wc_ego_edgelist(self,target_db,search_text, is_query_string=False, target_category='doc_subclass'):
#         es_query = ElasticQuery(target_db)
#         source = ['doc_id','doc_subclass','title','publication_date']
#         all_docs_generator = es_query.get_all_docs_by_query_string(search_text=search_text, source=source)
#         x = es_query.get_docs_by_search_text(search_text,size=10000, source=source)
#         edgelist_df = self._make_edgelist_from_es_docs_format(target_db, x, offsets=False, positions=False)
#         # chunk_size = 10000
#         # loop_index = True
#         # term_vector_df_list = []
#         # while loop_index:
#         #     docs_gen_list = []
#         #     try:
#         #         for idx in range(chunk_size):
#         #             docs_gen_list.append(next(all_docs_generator))
#         #         edgelist_df = self._make_edgelist_from_es_docs_format(target_db, docs_gen_list, offsets=False, positions=False)
#         #         term_vector_df_list.append(edgelist_df)
#         #     except StopIteration:
#         #         edgelist_df = self._make_edgelist_from_es_docs_format(target_db, docs_gen_list, offsets=False, positions=False)
#         #         term_vector_df_list.append(edgelist_df)
#         #         loop_index = False
#         # combiend_term_vector_df = pd.DataFrame()
#         # if len(term_vector_df_list)>0:
#         #     combiend_term_vector_df = pd.concat(term_vector_df_list, join='outer', ignore_index=True, axis=0)
#         #     combiend_term_vector_df['score'] = combiend_term_vector_df['score'].fillna(0)
#         # return combiend_term_vector_df


# if __name__ == '__main__':
#     import sys
#     import os
#     from pathlib import Path
#     APP_DIR = Path(__file__).resolve().parent.parent.parent
#     sys.path.insert(0, str(APP_DIR))
#     os.environ['DJANGO_SETTINGS_MODULE'] = 'config.settings.base'
#     os.environ['DJANGO_ALLOW_ASYNC_UNSAFE'] = 'true'
#     import django
#     django.setup()
#     from config.constants import WORDCOUNT_EDGELIST_COLUMN_MAPPING,SNA_EDGELIST_COLUMN_MAPPING, SUBJECT_PLAYER_EDGELIST_COLUMN_MAPPING
#     from db_manager.managers import ElasticQuery
#     import timeit
#     from terms.models import EdgelistTerm

#     start_time = timeit.default_timer()
#     target_db = 'docu_subject'
#     search_text = '인공지능'
#     query_string = "((검색* OR 경계블록* OR 계산* OR 단말* OR 데이터* OR 도어* OR 연결* OR 이용자* OR 인터넷* OR 자의* OR 지하* OR 차량* OR 충진* OR 콘크리트*) OR ((검색 OR 검색) AND (경계블록 OR 콘크리트) AND (계산 OR 계산) AND (단말 OR 연결 OR 도어 OR 지하) AND (데이터 OR 단말) AND (도어 OR 도어) AND (연결 OR 연결) AND (이용자 OR 이용자) AND (인터넷 OR 인터넷) AND (자의 OR 인터넷 OR 자의 OR 생년월일) AND (지하 OR 지하) AND (차량 OR 차량) AND (충진 OR 연결) AND (콘크리트 OR 콘크리트)))"

#     # res = es_query.tokenize_text('토크나이징 테스트입니ㅏ')
#     # edge = BaseEdgelist()
#     # all_edgelist = edge.get_all_edgelist(target_db=target_db, offsets=True)
#     # ego_edgelist = edge.get_ego_edgelist(target_db=target_db, search_text=search_text, offsets=True)
#     # refined_ego_edgelist =  edge.get_refined_ego_edgelist(target_db,search_text=search_text, offsets=True)
#     # print('\n', all_edgelist, '\n', ego_edgelist, '\n', refined_ego_edgelist, '\n')
#     # all_edgelist.to_parquet(f'/data/temp_parquet/{target_db}.parquet')
#     # print(refined_ego_edgelist)
#     # print(timeit.default_timer()-start_time)
#     # all_df = edge.get_all_edgelist(target_db, chunk_size=1000)

#     # start_time = timeit.default_timer()
#     # sna_edge = SnaEdgelist()
#     # ego_edgelist_df, refined_ego_edgelist = sna_edge.get_sna_refined_ego_edgelist(target_db=target_db, search_text=search_text)
#     # print(ego_edgelist_df.head(), refined_ego_edgelist.head())
#     # print(timeit.default_timer()-start_time)

#     # start_time = timeit.default_timer()
#     # wc_edge = WordCountEdgelist()
#     # wc_ego_edgelist = wc_edge.get_wc_ego_edgelist(target_db=target_db, search_text=search_text, target_category='doc_section', size=100)
#     # wc_ego_edgelist = wc_edge.get_wc_refined_ego_edgelist(target_db=target_db, search_text=search_text)
#     # print(wc_ego_edgelist.sort_values('doc_freq',ascending=False))
#     # print(timeit.default_timer()-start_time)

#     # start_time = timeit.default_timer()
#     # player_edge = PlayerEdgelist()
#     # ego_edgelist_df, refined_ego_edgelist = player_edge.get_custom_refined_ego_edgelist(target_db=target_db, search_text=search_text)
#     # print(ego_edgelist_df.head(), ego_edgelist_df.shape, refined_ego_edgelist.head(), refined_ego_edgelist.shape)
#     # print(timeit.default_timer()-start_time)

#     # edge = WordExtractorEdgelist()
#     # edge_df = edge.get_ego_edgelist(target_db=target_db, search_text=search_text, offsets=True, positions=True)
#     # print(edge_df)
    
#     # all_df = word_edge.get_all_edgelist(target_db)
#     # print(all_df)
#     # import datetime
#     # now = datetime.datetime.now().strftime('%Y%m%d%H%M')
#     # all_df.to_parquet(f'/data/temp_parquet/{target_db}_{now}.parquet')

#     edge = WordCountEdgelist()
#     x = edge.get_wc_ego_edgelist(target_db=target_db, search_text=query_string, is_query_string=True)
#     print(x)
#     x = edge.get_wc_refined_ego_edgelist(target_db=target_db, search_text=query_string, is_query_string=True)
#     print(x)

#     # start_time = timeit.default_timer()
#     # e = LimitedWordCountEdgelist()
#     # print(e.get_limited_wc_ego_edgelist(target_db, query_string, is_query_string=True))
#     # print(timeit.default_timer()-start_time)