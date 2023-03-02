import pandas as pd
from django.core.cache import cache

if __name__ !='__main__':
    from analysis.word_extractors.compound_extractor import EdgelistCompoundExtractor
    from db_manager.managers import ElasticQuery
    from terms.models import EdgelistTerm
    from config.constants import WORDCOUNT_EDGELIST_COLUMN_MAPPING, SNA_EDGELIST_COLUMN_MAPPING, SUBJECT_PLAYER_EDGELIST_COLUMN_MAPPING, BASE_EDGELIST_COLUMN_MAPPING



class BaseEdgelist():
    """
    문서-용어-빈도의 정보를 담고 있는 dataframe 생성 클래스
    처리 순서: 검색문장 -> 문서 검색 -> 문서 내 단어 elasticsearch 조회 -> 대표어,동의어,불용어 처리 -> 기술용어 외 단어 제거 -> 분석 모듈 요구 사항에 맞게 후처리
    """

    def _make_edgelist_from_es_docs_format(self, target_db, docs, offsets=False, positions=True):
        """
        elasticsearch 검색 결과에서 dataframe 으로 변환하는 공통 사용 함수 
        
        Parameters
        ----------
        target_db : str, 분석 대상 document index ex) docu_patent, docu_thesis, etc
        docs : list, elasticsearch search api 반환 결과 형태
            sample: [{'_index': 'docu_test', '_type': '_doc', '_id': '1019790004128', '_score': None, '_routing': 'A', '_source': {...}, 'sort': [...]},{..}]
        size : int, default = 500, 검색 문서 개수 (검색 score 상위 n 개)
        offsets : bool, default = False , 문서 내 offset(단어 시작 위치, 종료 위치) 반환
        positions : bool, default = True, 문서 내 position 정보 반환
        
        Returns
        -------
        edgelist_df : pandas.DataFrame
            df columns : 
                doc_id: str, 문서번호
                index: str, elasticsearch index 이름
                term: str, 단어
                field: str , 단어가 포함된 elasticsearch index mapping 필드
                doc_freq: int, shard 내 해당 단어 포함 문서 갯수
                ttf: int, shard 내 해당 단어의 전체 갯수
                term_freq: int, 문서 내 해당 단어의 갯수
                position: 문서 내 위치
                token_info: list(dict), 문서 내 offset 정보
                doc_section: str, routing 정보
                doc_subclass: str, doc_doc_subclass 정보
                score: float, 검색 시 단어의 score(bm25 기준)
        """
        es_query = ElasticQuery(target_db)
        docs_df = pd.DataFrame(docs)
        if docs_df.empty:
            return pd.DataFrame()
        for k,v in BASE_EDGELIST_COLUMN_MAPPING.items():
            docs_df[k] = docs_df['_source'].map(lambda x: x[v])
        docs_df = docs_df.set_index('_id')
        term_vectors = es_query.get_tokens_from_doc_info(docs, offsets=offsets, positions=positions)
        unpacked_term_vectors = es_query.unpacking_termsvector(term_vectors)
        edgelist_df = pd.DataFrame(unpacked_term_vectors)
        edgelist_df = edgelist_df.set_index(['doc_id'])
        edgelist_df['section'] = docs_df['_routing']
        edgelist_df['score'] = docs_df['_score']
        for k,v in BASE_EDGELIST_COLUMN_MAPPING.items():
            if k == 'doc_id':
                continue
            edgelist_df[k] = docs_df[k]
        edgelist_df = edgelist_df.reset_index()
        return edgelist_df

    def get_ego_edgelist(self, target_db, doc_ids, size=500, offsets=False, positions=True):
        """
        검색 문장에 해당하는 edgelist 생성

        Parameters
        ----------
        target_db : str, 분석 대상 document index ex) docu_patent, docu_thesis, etc
        search_text : str, 검색 문장, 단어 등 텍스트
        size : int, default = 500, 검색 문서 개수 (검색 score 상위 n 개)
        offsets : bool, default = False , 문서 내 offset(단어 시작 위치, 종료 위치) 반환
        positions : bool, default = True, 문서 내 position 정보 반환
        

        Returns
        -------
        edgelist_df : pandas.DataFrame
            _make_edgelist_from_es_docs_format 함수 return 값과 동일
        """
        es_query = ElasticQuery(target_db)
        source = list(BASE_EDGELIST_COLUMN_MAPPING.values())
        edgelist_df = pd.DataFrame()
        # query string 으로 edgelist 생성
        # if search_mode=='query':
        #     search_result = es_query.get_docs_by_full_query(query=search_value, doc_size=size)
        #     docs = search_result['hits']['hits']
        # elif search_mode=='search_text':
        #     docs = es_query.get_docs_by_search_text(search_text=search_value, size=size, source=source)
        # elif search_mode=='doc_ids':
        if not isinstance(doc_ids, list):
            return edgelist_df
        search_res = es_query.get_docs_by_doc_ids(doc_ids=doc_ids, source=source)
        docs = search_res['hits']['hits']
        edgelist_df = self._make_edgelist_from_es_docs_format(target_db, docs, offsets, positions)

        return edgelist_df

    def get_refined_ego_edgelist(self, target_db, doc_ids, size=500, offsets=False, positions=True, return_original=False):
        """
        Parameters
        ----------
        get_ego_edgelist 함수와 동일

        Returns
        -------
        edgelist_df : pandas.DataFrame
            _make_edgelist_from_es_docs_format 함수 return 값과 동일
        """
        edgelist_df = self.get_ego_edgelist(target_db, doc_ids=doc_ids, size=size, offsets=offsets, positions=positions)
        if edgelist_df.empty:
            if return_original:
                return edgelist_df, edgelist_df
            return edgelist_df
        # edgelist 용어 리스트 불러오기
        # target db edgelist 용어 (현재는 doc_subclass나 section 별로 용어를 구분하지 않고 있음)
        # start_time = timeit.default_timer()

        # 임시 캐시 미활성화
        # target_db_term = cache.get(f'{target_db}_edgelist_term')
        target_db_term = None
        if not target_db_term:
            target_db_term = set(EdgelistTerm.objects.using(target_db).all().values_list('term',flat=True))
            # cache.set(f'{target_db}_edgelist_term',target_db_term)
        
        # start_time = timeit.default_timer()
        ego_edgelist_df = edgelist_df[edgelist_df['term'].isin(target_db_term)]
        # 동의어/불용어/대표어 처리 or 엘라스틱서치 인덱싱 시 적용 추가 예정
        if return_original:
            return edgelist_df, ego_edgelist_df
        return ego_edgelist_df

    def get_all_edgelist(self, target_db, chunk_size=10000, offsets=False, positions=True):
        """
        target_db와 연결되어 있는 elasticsearch index 의 모든 문서의 단어/빈도수

        Parameters
        ----------
        target_db : str, 분석 대상 document index ex) docu_patent, docu_thesis, etc
        chunk_size : int, default = 10000, elasticsearch 에서 문서 내 단어를 반환 받을 때 한번의 request에 담을 문서 개수. 부하 방지위해 사용 
        offsets : bool, default = False , 문서 내 offset(단어 시작 위치, 종료 위치) 반환
        positions : bool, default = True, 문서 내 position 정보 반환
        
        Returns
        -------
        edgelist_df : pandas.DataFrame
            _make_edgelist_from_es_docs_format 함수 return 값과 동일
        """
        es_query = ElasticQuery(target_db)
        all_docs_generator = es_query.get_all_docs(source=list(BASE_EDGELIST_COLUMN_MAPPING.values()))
        loop_index = True
        term_vector_df_list = []
        while loop_index:
            docs_gen_list = []
            try:
                for idx in range(chunk_size):
                    docs_gen_list.append(next(all_docs_generator))
                edgelist_df = self._make_edgelist_from_es_docs_format(target_db, docs_gen_list, offsets, positions)
                term_vector_df_list.append(edgelist_df)
            except StopIteration:
                edgelist_df = self._make_edgelist_from_es_docs_format(target_db, docs_gen_list, offsets, positions)
                term_vector_df_list.append(edgelist_df)
                loop_index = False
        combiend_term_vector_df = pd.DataFrame()
        if len(term_vector_df_list)>0:
            combiend_term_vector_df = pd.concat(term_vector_df_list, join='outer', ignore_index=True, axis=0)
            combiend_term_vector_df['score'] = combiend_term_vector_df['score'].fillna(0)
        return combiend_term_vector_df


class SnaEdgelist(BaseEdgelist):
    """
    WordExtractor 분석 모듈에서 사용할 수 있는 형태로 edgelist 변환하는 클래스.
    EldgelistManager 로직 그대로 사용하고, 추가로 컬럼명 변환, title field의 position 정보 제거함
    """
    def __init__(self):
        super().__init__()
    
    def _transform(self, df):
        if df.empty:
            return df
        df = df.rename(SNA_EDGELIST_COLUMN_MAPPING, axis=1).drop('DELETE',axis=1)
        return df

    def get_all_edgelist(self, target_db, chunk_size=10000, offsets=False, positions=True):
        edgelist_df = super().get_all_edgelist(target_db, chunk_size=chunk_size, offsets=offsets, positions=positions)
        edgelist_df = self._transform(edgelist_df)
        return edgelist_df

    def get_sna_ego_edgelist(self, target_db, doc_ids, size=500, offsets=False, positions=True):
        ego_edgelist_df = self.get_ego_edgelist(target_db=target_db, doc_ids=doc_ids, size=size, offsets=offsets, positions=positions)
        ego_edgelist_df = self._transform(ego_edgelist_df)
        return ego_edgelist_df

    def get_sna_refined_ego_edgelist(self, target_db, doc_ids, size=500, offsets=False, positions=True, return_original=True, extract_compound=True):
        ego_edgelist_df, refined_ego_edgelist = self.get_refined_ego_edgelist(target_db=target_db, doc_ids=doc_ids, size=size, return_original=return_original)
        ego_edgelist_df = self._transform(ego_edgelist_df)
        #print("============================")
        #print("refined_ego_edgelist : ", refined_ego_edgelist)
        #print("============================")
        refined_ego_edgelist = self._transform(refined_ego_edgelist)
        #print("============================")
        #print("transform refined_ego_edgelist : ", refined_ego_edgelist)
        #print("============================")
        if extract_compound and not refined_ego_edgelist.empty:
            compound_extractor = EdgelistCompoundExtractor(refined_ego_edgelist=refined_ego_edgelist)
            comp_df = compound_extractor.get_compounds_df()
            if not comp_df.empty:
                attach_col = list(set(refined_ego_edgelist.columns)-set(comp_df.columns))
                comp_df = comp_df.set_index('doc_id')
                comp_df[attach_col] = refined_ego_edgelist.drop_duplicates('doc_id').set_index('doc_id')[attach_col]
                comp_df = comp_df.reset_index()
                refined_ego_edgelist = pd.concat([refined_ego_edgelist,comp_df],axis=0)
        if return_original:
            return ego_edgelist_df, refined_ego_edgelist
        return refined_ego_edgelist

class PlayerEdgelist(BaseEdgelist):
    """
    Player 네트워크 분석 모듈에서 사용할 수 있는 형태로 edgelist 변환하는 클래스.
    EldgelistManager 로직 그대로 사용하고, 추가로 Player 정보를 추가함
    """
    def __init__(self):
        super().__init__()

    def _make_player_df_from_es_docs_format(self, docs, source_fields):
        """
        elasticsearch player 검색 결과에서 dataframe 으로 변환
        
        Parameters
        ----------
        docs : list, elasticsearch search api 반환 결과 형태
            sample: [{'_index': 'docu_test', '_type': '_doc', '_id': '1019790004128', '_score': None, '_routing': 'A', '_source': {...}},{..}]
        
        Returns
        -------
        edgelist_df : pandas.DataFrame
            df columns : 
                given source_fields parameters
        """
        docs_df = pd.DataFrame(docs)
        if docs_df.empty:
            return pd.DataFrame()
        player_df = pd.DataFrame()
        for source_field in source_fields:
            player_df[source_field] = docs_df['_source'].map(lambda x: x[source_field])
        return player_df

    def _transform(self, df):
        if df.empty:
            return df
        df = df.rename(SUBJECT_PLAYER_EDGELIST_COLUMN_MAPPING, axis=1).drop('DELETE',axis=1)
        return df

    def get_player_ego_edgelist(self, target_db, doc_ids, size=500, offsets=False, positions=False):
        ego_edgelist_df = self.get_ego_edgelist(target_db=target_db, doc_ids=doc_ids, size=size, offsets=offsets, positions=positions)
        ego_edgelist_df = self._transform(ego_edgelist_df)
        return ego_edgelist_df

    def get_player_refined_ego_edgelist(self, target_db, doc_ids, size=500, offsets=False, positions=False, return_original=True):
        ego_edgelist_df, refined_ego_edgelist = self.get_refined_ego_edgelist(target_db=target_db, doc_ids=doc_ids, size=size, return_original=True)
        if refined_ego_edgelist.empty:
            if return_original:
                return ego_edgelist_df, refined_ego_edgelist
            return refined_ego_edgelist
        ego_edgelist_df = self._transform(ego_edgelist_df)
        refined_ego_edgelist = self._transform(refined_ego_edgelist)
        if return_original:
            return ego_edgelist_df, refined_ego_edgelist
        return refined_ego_edgelist


# class __WordCountEdgelist(BaseEdgelist):
#     """
#     """
#     def __init__(self):
#         super().__init__()
    
#     def _transform(self, df):
#         df = df.rename(WORDCOUNT_EDGELIST_COLUMN_MAPPING, axis=1).drop('DELETE',axis=1)
#         return df

#     def get_term_doc_id_mapping_from_edgelist(self, df):
#         try:
#             df = df[['term','doc_id','publication_date','subclass']]
#             df = df.drop_duplicates()
#             term_doc_id_mapping = df.groupby('term')[['doc_id','publication_date','subclass']].apply(lambda x: {'doc_id':list(x['doc_id']),'pyear':list(x['publication_date']),'subclass':list(x['subclass'])}).to_dict()
#             return term_doc_id_mapping 
#         except KeyError:
#             return {}

#     def get_wc_ego_edgelist(self, target_db, doc_ids, size=500, target_category='doc_subclass', offsets=False, positions=False):
#         es_query = ElasticQuery(target_db)

#         ego_edgelist_df = self.get_ego_edgelist(target_db=target_db, doc_ids=doc_ids, size=size, offsets=offsets, positions=positions)
#         term_doc_id_mapping = self.get_term_doc_id_mapping_from_edgelist(ego_edgelist_df)
#         if ego_edgelist_df.empty:
#             return ego_edgelist_df
#         ego_edgelist_df = self._transform(ego_edgelist_df)
#         ego_edgelist_df = ego_edgelist_df.drop_duplicates()
#         yearly_category_doc_freq_list = es_query.calculate_yearly_category_doc_freq(ego_edgelist_df.to_dict('records'), target_category=target_category)
#         ego_edgelist_df = pd.DataFrame(yearly_category_doc_freq_list)
#         return ego_edgelist_df, term_doc_id_mapping

#     def get_wc_refined_ego_edgelist(self, target_db, doc_ids, size=500, target_category='doc_subclass', offsets=False, positions=False):
#         es_query = ElasticQuery(target_db)
        
#         refined_ego_edgelist = self.get_refined_ego_edgelist(target_db=target_db, doc_ids=doc_ids, size=size, offsets=offsets, positions=positions)
#         term_doc_id_mapping = {}
#         if refined_ego_edgelist.empty:
#             return refined_ego_edgelist, term_doc_id_mapping
#         refined_ego_edgelist = self._transform(refined_ego_edgelist)
#         refined_ego_edgelist = refined_ego_edgelist.drop_duplicates()
#         yearly_category_doc_freq_list = es_query.calculate_yearly_category_doc_freq(refined_ego_edgelist.to_dict('records'), target_category=target_category)
#         term_doc_id_mapping = self.get_term_doc_id_mapping_from_edgelist(refined_ego_edgelist)
#         refined_ego_edgelist = pd.DataFrame(yearly_category_doc_freq_list)
#         return refined_ego_edgelist, term_doc_id_mapping

class WordCountEdgelist(BaseEdgelist):
    """
    """
    def __init__(self):
        super().__init__()
    
    def _transform(self, df):
        df = df.rename(WORDCOUNT_EDGELIST_COLUMN_MAPPING, axis=1).drop('DELETE',axis=1)
        return df

    # def get_term_doc_id_mapping_from_edgelist(self, df):
    #     try:
    #         df = df[['term','doc_id','publication_date','subclass']]
    #         df = df.drop_duplicates()
    #         term_doc_id_mapping = df.groupby('term')[['doc_id','publication_date','subclass']].apply(lambda x: {'doc_id':list(x['doc_id']),'pyear':list(x['publication_date']),'subclass':list(x['subclass'])}).to_dict()
    #         return term_doc_id_mapping 
    #     except KeyError:
    #         return {}

    def get_wc_ego_edgelist(self, target_db, doc_ids, size=500, offsets=False, positions=False):
        ego_edgelist_df = self.get_ego_edgelist(target_db=target_db, doc_ids=doc_ids, size=size, offsets=offsets, positions=positions)
        if ego_edgelist_df.empty:
            return ego_edgelist_df
        ego_edgelist_df = self._transform(ego_edgelist_df)
        ego_edgelist_df['ipr'] = ego_edgelist_df['ipr'].map(len)
        ego_edgelist_df['paper'] = ego_edgelist_df['paper'].map(len)
        return ego_edgelist_df

    def get_wc_refined_ego_edgelist(self, target_db, doc_ids, size=500, offsets=False, positions=False):
        refined_ego_edgelist = self.get_refined_ego_edgelist(target_db=target_db, doc_ids=doc_ids, size=size, offsets=offsets, positions=positions)
        if refined_ego_edgelist.empty:
            return refined_ego_edgelist
        refined_ego_edgelist = self._transform(refined_ego_edgelist)
        refined_ego_edgelist['ipr'] = refined_ego_edgelist['ipr'].map(len)
        refined_ego_edgelist['paper'] = refined_ego_edgelist['paper'].map(len)
        return refined_ego_edgelist


if __name__ == '__main__':
    import sys
    import os
    from pathlib import Path
    APP_DIR = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(APP_DIR))
    os.environ['DJANGO_SETTINGS_MODULE'] = 'config.settings.base'
    os.environ['DJANGO_ALLOW_ASYNC_UNSAFE'] = 'true'
    import django
    django.setup()
    from analysis.word_extractors.compound_extractor import EdgelistCompoundExtractor
    from config.constants import WORDCOUNT_EDGELIST_COLUMN_MAPPING,SNA_EDGELIST_COLUMN_MAPPING, SUBJECT_PLAYER_EDGELIST_COLUMN_MAPPING, BASE_EDGELIST_COLUMN_MAPPING
    from db_manager.managers import ElasticQuery
    import timeit
    from terms.models import EdgelistTerm
    target_db='kistep_sbjt'
    es_query = ElasticQuery(target_db=target_db)
    search_form = {
        "search_text" : ["인공지능"]
    }
    doc_size = 1000
    from documents.document_utils import create_keyword_search_query
    query = create_keyword_search_query(search_form=search_form, size=doc_size)
    search_result = es_query.get_docs_by_full_query(query=query, doc_size=doc_size)
    doc_ids_score = {x['_id']:x['_score'] for x in search_result['hits']['hits']}
    doc_ids = list(doc_ids_score.keys())
    doc_size = len(doc_ids)
    # p_edge = PlayerEdgelist()
    # ego_edgelist_df, refined_ego_edgelist= p_edge.get_player_refined_ego_edgelist(target_db, doc_ids, size=doc_size, offsets=False, positions=True, return_original=True)
    wc = WordCountEdgelist()
    refined_ego_edgelist = wc.get_wc_refined_ego_edgelist(doc_ids=doc_ids,target_db=target_db)
    print(refined_ego_edgelist.head(), refined_ego_edgelist.shape)