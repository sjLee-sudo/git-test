from django.conf import settings
from rest_framework.decorators import api_view
from rest_framework.response import Response
from elasticsearch.exceptions import RequestError
from drf_yasg.utils import swagger_auto_schema
from rest_framework import status
from drf_yasg import openapi 
import pandas as pd
from utils.custom_errors import APPLY_400_INVALID_DOC_SIZE, APPLY_400_SEARCH_FORM_REQUIRED, APPLY_400_INVALID_ANALYSIS_TYPE, APPLY_400_INVALID_QUERY_STRING, APPLY_400_DOC_IDS_REQUIRED, APPLY_400_FULL_QUERY_REQUIRED, FailedQueryParsing, TooManyClause
from config.constants import SUPPORT_ANALYSIS_TYPES, TARGET_DB, ES_DOC_SOURCE, CHART_FUNCTION_MAPPING, DOC_SIZE_LIMIT
from documents.models import Category
from documents.document_utils import create_keyword_search_query, create_subject_search_query, create_topic_model_search_query
from analysis.views import CHART_FUNCTION_SELECTOR, create_api_format_for_anlaysis
from analysis.edgelist.kistep_edgelist import SnaEdgelist, WordCountEdgelist, PlayerEdgelist
from analysis.sna.sna import extract_topic_word, player_network, orgn_network
from analysis.word_extractors.topic_extractor import extract_recommend_word
from analysis.sna.lda import lda_fullprocess
from db_manager.managers import ElasticQuery
import json
import timeit
from django.utils import timezone

def attach_sna_sort_index(sna_result, doc_data):
    paper_sort_df = pd.DataFrame(sna_result.get('paper_sort_index'))
    if paper_sort_df is not None and not paper_sort_df.empty  :
        paper_sort_df = paper_sort_df.set_index('doc_id')
        paper_df = pd.DataFrame(doc_data)
        paper_df = paper_df.set_index('_id')
        # sort 용 지표 생성
        paper_df[['_rltd','_infl','_btwn','_degr']] = paper_sort_df[['S','E','B','D']]
        paper_df[['_rltd','_infl','_btwn','_degr']] = paper_df[['_rltd','_infl','_btwn','_degr']].fillna(0).round(2)
        paper_df['_rcnt'] = paper_df['_source'].map(lambda x: x['stan_yr'])
        doc_data = paper_df.reset_index().fillna("").to_dict('records')
    return doc_data

def get_analysis_response(search_result, target_analysis_list, include_doc_data=True, search_form=None, required_doc_size=None, use_cache=True):
    es_query = ElasticQuery(target_db=TARGET_DB)
    response_format = {'data': {'total':0,'doc_ids':[],'doc_size':0,'doc_data':[], 'analysis_data':{}}}

    # doc_ids = [x['_id'] for x in search_result['hits']['hits']]
    if required_doc_size:
        doc_ids_score = {x['_id']: 0 if x['_score'] is None else x['_score'] for x in search_result['hits']['hits'][:required_doc_size]}
        doc_ids_title = {x['_id']: '' if x['_source']['kor_pjt_nm'] is None else x['_source']['kor_pjt_nm'] for x in search_result['hits']['hits'][:required_doc_size]}
        doc_data = search_result['hits']['hits'][:required_doc_size]
    else:
        doc_ids_score = {x['_id']: 0 if x['_score'] is None else x['_score'] for x in search_result['hits']['hits']}
        doc_ids_title = {x['_id']: '' if x['_source']['kor_pjt_nm'] is None else x['_source']['kor_pjt_nm'] for x in search_result['hits']['hits']}
        doc_data = search_result['hits']['hits']
    doc_ids = list(doc_ids_score.keys())
    sorted_doc_ids = ','.join(sorted(doc_ids))
    
    doc_size = len(doc_ids)
    response_format['data'].update({'doc_ids':doc_ids})
    response_format['data'].update({'total':search_result['hits']['total']['value']})
    response_format['data'].update({'doc_size':doc_size})
    if doc_size == 0:
        return response_format
    
    non_cache_target_analysis_list = []

    # print('\033[95m' + "=========================================================" + '\033[0m')
    # print('\033[95m' + " cache check start >> " + str(timezone.now()) + '\033[0m')
    # print('\033[95m' + "=========================================================" + '\033[0m')
    # check cache
    for target_analysis in target_analysis_list:
        if use_cache:
            cache_data = es_query.get_analysis_cache_by_search_keys(target_db=TARGET_DB, search_keys=sorted_doc_ids, analysis_type=target_analysis, doc_size=doc_size)
            if cache_data:
                response_format['data']['analysis_data'].update(cache_data['_source']['analysis_result'][target_analysis])
                if target_analysis == 'sna':
                    doc_data = attach_sna_sort_index(sna_result=cache_data['_source']['analysis_result']['sna'],doc_data=doc_data)
                continue
        non_cache_target_analysis_list.append(target_analysis)
    # print('\033[95m' + "=========================================================" + '\033[0m')
    # print('\033[95m' + " cache check end >> " + str(timezone.now()) + '\033[0m')
    # print('\033[95m' + "=========================================================" + '\033[0m')
    #
    # print('\033[95m' + "=========================================================" + '\033[0m')
    # print('\033[95m' + " create edgelist start >> " + str(timezone.now()) + '\033[0m')
    # print('\033[95m' + "=========================================================" + '\033[0m')
    # edgelist prepare
    sna_ego_edge = pd.DataFrame()
    sna_refined_ego_edge = pd.DataFrame()
    player_refined_ego_edge = pd.DataFrame()
    wordcount_refined_ego_edge = pd.DataFrame()
    # edgelist prepare
    for target_analysis in non_cache_target_analysis_list:
        # print('\033[95m' + "=========================================================" + '\033[0m')
        # print('\033[95m' + target_analysis+"edgelist start >> " + str(timezone.now()) + '\033[0m')
        # print('\033[95m' + "=========================================================" + '\033[0m')
        if target_analysis in ['sna', 'lda','rec_kwd']:
            if sna_ego_edge.empty and sna_refined_ego_edge.empty:
                sna_edge = SnaEdgelist()
                sna_ego_edge, sna_refined_ego_edge = sna_edge.get_sna_refined_ego_edgelist(target_db=TARGET_DB, doc_ids=doc_ids, size=doc_size, extract_compound=True)
        if target_analysis in ['player_sna']:
            if player_refined_ego_edge.empty:
                player_edge = PlayerEdgelist()
                player_refined_ego_edge = player_edge.get_player_refined_ego_edgelist(target_db=TARGET_DB, doc_ids=doc_ids, size=doc_size, return_original=False)
        if target_analysis in CHART_FUNCTION_MAPPING.keys():
            if wordcount_refined_ego_edge.empty:
                wc_edgelist = WordCountEdgelist()
                wordcount_refined_ego_edge = wc_edgelist.get_wc_refined_ego_edgelist(target_db=TARGET_DB, doc_ids=doc_ids, size=doc_size, offsets=False, positions=False)
    #     print('\033[95m' + "=========================================================" + '\033[0m')
    #     print('\033[95m' + target_analysis + "edgelist end >> " + str(timezone.now()) + '\033[0m')
    #     print('\033[95m' + "=========================================================" + '\033[0m')
    # print('\033[95m' + "=========================================================" + '\033[0m')
    # print('\033[95m' + " create edgelist end >> " + str(timezone.now()) + '\033[0m')
    # print('\033[95m' + "=========================================================" + '\033[0m')

    # print('\033[95m' + "=========================================================" + '\033[0m')
    # print('\033[95m' + " analysis start >> " + str(timezone.now()) + '\033[0m')
    # print('\033[95m' + "=========================================================" + '\033[0m')
    # update analysis result
    for target_analysis in non_cache_target_analysis_list:
        analysis_res = {}
        # print('\033[95m' + "=========================================================" + '\033[0m')
        # print('\033[95m' + target_analysis + " start >> " + str(timezone.now()) + '\033[0m')
        # print('\033[95m' + "=========================================================" + '\033[0m')
        if target_analysis in ['sna', 'lda','rec_kwd']:
            analysis_res = analysis_job_controller(target_analysis=target_analysis, ego_edge=sna_ego_edge, refined_ego_edge=sna_refined_ego_edge, doc_ids_score=doc_ids_score, search_form=search_form, doc_ids_title=doc_ids_title)
        if target_analysis in ['player_sna']:
            analysis_res = analysis_job_controller(target_analysis=target_analysis, refined_ego_edge=player_refined_ego_edge)
        if target_analysis in CHART_FUNCTION_MAPPING.keys():
            analysis_res = analysis_job_controller(target_analysis=target_analysis, refined_ego_edge=wordcount_refined_ego_edge)
        # 추가 작업
        ## 정렬 지표 doc data에 추가
        if target_analysis == 'sna':
            doc_data = attach_sna_sort_index(sna_result=analysis_res, doc_data=doc_data)

        # print('\033[95m' + "=========================================================" + '\033[0m')
        # print('\033[95m' + target_analysis + " end >> " + str(timezone.now()) + '\033[0m')
        # print('\033[95m' + "=========================================================" + '\033[0m')
        response_format['data']['analysis_data'].update(analysis_res)

        # insert cache
        es_query.insert_analysis_cache(target_db=TARGET_DB, search_keys=sorted_doc_ids, analysis_type=target_analysis, doc_size=doc_size, analysis_result={target_analysis:analysis_res})
    # print('\033[95m' + "=========================================================" + '\033[0m')
    # print('\033[95m' + " analysis end >> " + str(timezone.now()) + '\033[0m')
    # print('\033[95m' + "=========================================================" + '\033[0m')
    if include_doc_data:
        # doc_data 에 score 값 source에 추가 (이력 데이터용)
        for doc in doc_data:
            doc['_source'].update({'rltd': 0 if not doc['_score'] else doc['_score']})
            doc['_source'].update({'ipr_cnt': len(doc['_source'].pop('ipr'))})
            doc['_source'].update({'paper_cnt': len(doc['_source'].pop('paper'))})
        response_format['data'].update({'doc_data': doc_data})
    return response_format
    
def analysis_job_controller(target_analysis, ego_edge=pd.DataFrame(), refined_ego_edge=pd.DataFrame(), doc_ids_score={}, search_form=None, doc_ids_title={}):
    analysis_res = {}
    if target_analysis =='sna':
        # sna analysis
        word_json = {}
        paper_json = {}
        paper_cent = pd.DataFrame()
        word_paper_map = {}
        compound_paper_map = {}
        paper_rank_list = []
        if not ego_edge.empty and not refined_ego_edge.empty:
            word_json_res, paper_json_res, paper_cent_res = extract_topic_word(ego_edge, refined_ego_edge, doc_ids_score)
            if word_json_res is not None and paper_json_res is not None and paper_cent_res is not None:
                word_json = word_json_res
                paper_json = paper_json_res
                paper_cent = paper_cent_res
                # rank 선정 기준 확인
                paper_rank = paper_cent_res.sort_values(['S'],ascending=False)
                paper_rank['kor_pjt_nm'] = paper_rank.doc_id.map(doc_ids_title)
                paper_rank_list = paper_rank[['doc_id','kor_pjt_nm']].to_dict('records')
            compound_df = refined_ego_edge[refined_ego_edge.col_type=='compound'][['word','doc_id']]
            if not compound_df.empty:
                compound_paper_map = compound_df.groupby('word')['doc_id'].agg(list).to_dict()
            word_paper_map = ego_edge.groupby('word')['doc_id'].agg(list).to_dict()
            word_paper_map.update(compound_paper_map)
        analysis_res.update({'sna':{'word_sna':word_json, 'paper_sna':paper_json},'word_paper_map':word_paper_map, 'paper_sort_index':paper_cent.to_dict(), 'sbjt_rank': paper_rank_list})
    elif target_analysis == 'lda':
        lda_result = ''
        # lda analysis
        if not ego_edge.empty and not refined_ego_edge.empty and search_form and 'search_text' in search_form and len(search_form['search_text']) >0:
            target_keyword = search_form['search_text'][0]
            lda_result = lda_fullprocess(search_text=target_keyword, _edgelist = refined_ego_edge, s_dict = doc_ids_score, remove_n = 2, no_below = 2, no_above = 1, _start = 3, _end = 10, p_limit = 0.1)
        analysis_res.update({'lda':lda_result})
    elif target_analysis == 'player_sna':
        player_sna_res = {'nodes':[], 'links':[], 'player_rank':[]}
        if not refined_ego_edge.empty:
            
            orgn_sna_res = orgn_network(refined_ego_edge, _tree=True)
            orgn_df = pd.DataFrame(orgn_sna_res['nodes'])
            if not orgn_df.empty:
                orgn_df = orgn_df[(orgn_df.section.isna()==False)&(orgn_df.t!='데이터미수집기관')]
                top_rank_orgn_df = orgn_df[['t','doc_id']]
                top_rank_orgn_df['doc_size'] = top_rank_orgn_df.doc_id.map(len)
                top_rank_orgn_df = top_rank_orgn_df.sort_values('doc_size',ascending=False)
                top_rank_orgn_df = top_rank_orgn_df.rename({'t':'name'},axis=1)
                orgn_rank_list = top_rank_orgn_df.to_dict('records')
                orgn_sna_res.update({'player_rank': orgn_rank_list})
                
            
            player_sna_res = player_network(refined_ego_edge, _tree=True)
            # 보유 과제 수별로 정렬
            player_df = pd.DataFrame(player_sna_res['nodes'])
            if not player_df.empty:
                player_df = player_df[(player_df.section.isna()==False)&(player_df.t!='데이터미수집기관')]
                top_rank_player_df = player_df[['t','doc_id']]
                top_rank_player_df['doc_size'] = top_rank_player_df.doc_id.map(len)
                top_rank_player_df = top_rank_player_df.sort_values('doc_size',ascending=False)
                top_rank_player_df = top_rank_player_df.rename({'t':'name'},axis=1)
                player_rank_list = top_rank_player_df.to_dict('records')
                player_sna_res.update({'player_rank': player_rank_list})
            sna_reslt=dict(rsrchr_sna = player_sna_res,orgn_sna=orgn_sna_res)
            
        analysis_res.update({'player_sna': sna_reslt})
    elif target_analysis == 'rec_kwd':
        compounds = extract_recommend_word(refined_ego_edge)
        # compounds = EdgelistCompoundExtractor(refined_ego_edgelist=refined_ego_edge).get_compounds(limit=100)
        # rec_kwd = [comp.replace(" ","") for comp in compounds]
        analysis_res.update({'rec_kwd':compounds})
    elif target_analysis in CHART_FUNCTION_MAPPING.keys():
        chart_result = {}
        if not refined_ego_edge.empty:
            if target_analysis == 'bar_chart':
                SECTION_CD_NM_MAP = dict(Category.objects.using(TARGET_DB).values_list('section_cd','section_nm').distinct())
                chart_result = CHART_FUNCTION_SELECTOR[target_analysis](refined_ego_edge, category_map=SECTION_CD_NM_MAP, n=10 )
            else:
                chart_result = CHART_FUNCTION_SELECTOR[target_analysis](refined_ego_edge)
        analysis_res.update({target_analysis: chart_result})
    return analysis_res


def get_search_result_by_full_query(query, filter_percent=10):
    es_query = ElasticQuery(target_db=TARGET_DB)
    # valid_result = es_query.validate_query_string(query_string=query)
    # if not valid_result['valid']:
    #     raise FailedQueryParsing(valid_result['error'])
    min_score = 0
    try:
        # min_score = es_query.get_percentile_score(query['query'],filter_percent=filter_percent)
        query.update({'min_score': min_score})
        query.update({'_source':ES_DOC_SOURCE})
        search_result = es_query.get_docs_by_full_query(query=query, doc_size=DOC_SIZE_LIMIT)
        # search_result = es_query.scroll_docs_by_query(query=query['query'], source=ES_DOC_SOURCE, min_score=min_score)
        return search_result
    except RequestError:
        raise FailedQueryParsing(f'failed to parse querystring : {query["query"]}')

body_query_string = openapi.Schema(
    type=openapi.TYPE_OBJECT,
    required=['search_form'],
    example=create_api_format_for_anlaysis(target=['search_form'])['example'],
    properties=create_api_format_for_anlaysis(target=['search_form'],doc_size=300)['properties'],
)
@swagger_auto_schema(
    method='post', 
    request_body=body_query_string,
    responses={
    200 : '토픽모델 검색 결과 데이터',
    400 : 
        APPLY_400_SEARCH_FORM_REQUIRED.as_md() +
        APPLY_400_INVALID_DOC_SIZE.as_md() +
        APPLY_400_INVALID_ANALYSIS_TYPE.as_md(),
    }
)
@api_view(['POST'])
def get_docs_by_query_string(request):
    """
    검색식을 통한 문서 조회
    ---
    Request
    ---
    ```python
    {
        "target_analysis_list": type: list(str) | required | comment: ["분석모듈명1","분석모듈명2"],
        "doc_size": type: int | default: None | comment:조회/분석할 문서 수 제한 (deprecated, 빠른 응답을 위한 테스트 용도로만 사용할 것),
        "include_doc_data" : type:boolean | default: true | comment:반환값에 문서데이터 첨부여부. 분석결과만 필요 시 false,
        "use_cache": boolean | default: true | comment: 미리 분석된 캐시데이터가 있다면 사용
        "search_form": {
            "query_string": "검색 쿼리",
            "filter": [{"필터링할필드1":["필터값1","필터값2"]},{"필터링할필드2":["필터값1","필터값2"]}]
        }
    }        
    ```
    Response
    ---
    ```python
    {
        "full_query": "검색에 사용된 쿼리, 시각화 요청 시 인자값으로 전달",
        "data": [
            "doc_ids": ["문서id1",],
            "total": "검색된 전체 문서수",
            "doc_size": "제공 문서수",
            "doc_data": [
                {
                    "_index": target_db 인덱스명,
                    "_type": 엘라스틱서치 인덱스 타입 (_doc으로 고정되어 있음),
                    "_id": "문서id",
                    "_score": "검색 결과 스코어",
                    "_rltd":"키워드 관련성",
                    "_infl":"과제 영향력",
                    "_btwn":"과제간 매개력",
                    "_degr":"과제 연결성",
                    "_rcnt":"과제 연도"
                    "_routing": "문서 소속 shard 내 routing 경로",
                    "_source": {
                        'doc_id':'문서id(과제고유번호와 동일)', 
                        'pjt_id':'과제고유번호', 'stan_yr':'기준년도', yr_cnt: '연차','hm_nm':'연구책임자명',
                        'rsch_area_cls_cd':'국가과학기술표준분류코드','rsch_area_cls_nm':'국가과학기술표준분류명',
                        'kor_kywd':'한글키워드', 'eng_kywd':'영문키워드', 
                        'rsch_goal_abstract':'연구목표요약', 'rsch_abstract':'연구내용요약', 'exp_efct_abstract':'기대효과요약', 
                        'pjt_prfrm_org_cd':'과제수행기관코드', 'pjt_prfrm_org_nm':'과제수행기관명',
                        'kor_pjt_nm':'국문과제명', 'eng_pjt_nm':'영문과제명', 
                        'pjt_mgnt_org_cd':'과제관리전문기관코드', 'spclty_org_nm':'과제관리전문기관명', 
                        'prctuse_nm':'실용화대상여부', 'rnd_phase':'연구개발단계', 'rsch_exec_suj':'연구수행주체', 'dtl_pjt_clas':'연구개발과제성격',
                        'tech_lifecyc_nm':'기술수명주기', 'regn_nm':'지역', 'pjt_no':'(기관)세부과제번호',
                        'tot_rsch_start_dt':'총연구기간시작일자', 'tot_rsch_end_dt':'총연구기간종료일자', 
                        'tsyr_rsch_start_dt':'당해년도연구시작일자','tsyr_rsch_end_dt':'당해년도연구종료일자',
                        'rndco_tot_amt':'연구비합계금액',
                    }
                },
            ],
            "analysis_data": {
                "분석명": "분석결과"
            }
        ]
    }
    ```
    Analysis List
    ---
    지원 분석 목록
    - sna: (문서/용어) 네트워크분석 /analysis/sna
    - rec_kwd: 단어추천 목록  /analysis/rec_kwd
    - wordcloud: 워드클라우드 /analysis/wordcloud

    Example
    ---
    ```python
    {
        "target_analysis_list": ["sna","wordcloud","rec_kwd"],
        "include_doc_data" : true,
        "doc_size": 10,
        "search_form": {
            "query_string": "(인공지능 | 빅데이터) + -(조류 | 독감)",
            "filter": [{"doc_section":["LA","UK"]}]
        }
    }
    ```
    """
    # 검색식 검색
    target_db = TARGET_DB
    doc_size = request.data.get('doc_size',None)
    search_form = request.data.get('search_form',None)
    target_analysis_list = request.data.get('target_analysis_list', [])
    include_doc_data = request.data.get('include_doc_data',True)
    filter_percent = request.data.get('filter_percent', 10)
    use_cache = request.data.get('use_cache',True)

    start_time = timeit.default_timer()
    # validation
    # if doc_size > 1001:
    #     return Response(status=APPLY_400_INVALID_DOC_SIZE.status, data={'error':APPLY_400_INVALID_DOC_SIZE.message})
    if not search_form:
        return Response(status=APPLY_400_SEARCH_FORM_REQUIRED.status, data={'error': APPLY_400_SEARCH_FORM_REQUIRED.message})
    for target_analysis in target_analysis_list:
        if not target_analysis or not target_analysis in SUPPORT_ANALYSIS_TYPES  or target_analysis == 'lda':
            return Response(status=APPLY_400_INVALID_ANALYSIS_TYPE.status, data={'error':APPLY_400_INVALID_ANALYSIS_TYPE.message})            
    if not 'query_string' in search_form or not search_form['query_string']:
        return Response(status=APPLY_400_INVALID_QUERY_STRING.status, data={'error': APPLY_400_INVALID_QUERY_STRING.message})
    es_query = ElasticQuery(target_db=target_db)
    # query validate
    query_string = search_form['query_string']
    valid_result = es_query.validate_query_string(query_string=query_string)
    if not valid_result['valid']:
        return Response(status=APPLY_400_INVALID_QUERY_STRING.status, data={'error': valid_result['error']})
    try:
        query = create_topic_model_search_query(search_form=search_form, use_simple_query=False)
        search_result = get_search_result_by_full_query(query=query, filter_percent=filter_percent)
        response_format = get_analysis_response(search_result=search_result, target_analysis_list=target_analysis_list, include_doc_data=include_doc_data, required_doc_size=doc_size, use_cache=use_cache)
        parsed_full_query = json.dumps(query, ensure_ascii=False)
        response_format.update({'full_query':parsed_full_query})
        return Response(status=status.HTTP_200_OK, data=response_format)
    except FailedQueryParsing as fqp:
        return Response(status=status.HTTP_500_INTERNAL_SERVER_ERROR, data={'error': str(fqp)})
    except TooManyClause as tmc:
        return Response(status=status.HTTP_400_BAD_REQUEST, data={'error': str(tmc)})        
    except RequestError as e: 
        return Response(status=status.HTTP_400_BAD_REQUEST, data={'error': str(e)})
    finally:
        if settings.DEBUG:
            print('analysis_target:',target_analysis_list,'took',timeit.default_timer()-start_time)

body_query_string = openapi.Schema(
    type=openapi.TYPE_OBJECT,
    required=['search_form'],
    example=create_api_format_for_anlaysis(target=['search_form'])['example'],
    properties=create_api_format_for_anlaysis(target=['search_form'],doc_size=300)['properties'],
)
@swagger_auto_schema(
    method='post', 
    request_body=body_query_string,
    responses={
    200 : '키워드 검색 결과 데이터',
    400 : 
        APPLY_400_SEARCH_FORM_REQUIRED.as_md() +
        APPLY_400_INVALID_DOC_SIZE.as_md() +
        APPLY_400_INVALID_ANALYSIS_TYPE.as_md(),
    }
)
@api_view(['POST'])
def get_docs_by_keyword(request):
    """
    키워드/문장을 통한 문서 조회
    ---
    Request
    ---
    ```python
    {
        "target_analysis_list": type: list(str) | required | comment: ["분석모듈명1","분석모듈명2"],
        "doc_size": type: int | default: None | comment:조회/분석할 문서 수 제한 (deprecated, 빠른 응답을 위한 테스트 용도로만 사용할 것),
        "include_doc_data" : type:boolean | default: true | comment:반환값에 문서데이터 첨부여부. 분석결과만 필요 시 false,
        "use_cache": boolean | default: true | comment: 미리 분석된 캐시데이터가 있다면 사용
        "search_form" : {
            "search_text": ['텍스트1','텍스트2'],
            "ordered_text": ['우선순위1','우선순위2'],
            "exclude_text": ['제외단어1','제외단어2'],
            "filter": [{"필터링할필드":["필터값1","필터값2"]}]
        }
    }
    ```
    Response
    ---
    ```python
    {
        "full_query": "검색에 사용된 쿼리, 시각화 요청 시 인자값으로 전달",
        "data": [
            "doc_ids": ["문서id1",],
            "total": "검색된 전체 문서수",
            "doc_size": "제공 문서수",
            "doc_data": [
                {
                    "_index": target_db 인덱스명,
                    "_type": 엘라스틱서치 인덱스 타입 (_doc으로 고정되어 있음),
                    "_id": "문서id",
                    "_score": "검색 결과 스코어",
                    "_rltd":"키워드 관련성",
                    "_infl":"과제 영향력",
                    "_btwn":"과제간 매개력",
                    "_degr":"과제 연결성",
                    "_rcnt":"과제 연도"
                    "_routing": "문서 소속 shard 내 routing 경로",
                    "_source": {
                        'doc_id':'문서id(과제고유번호와 동일)', 
                        'pjt_id':'과제고유번호', 'stan_yr':'기준년도', yr_cnt: '연차','hm_nm':'연구책임자명',
                        'rsch_area_cls_cd':'국가과학기술표준분류코드','rsch_area_cls_nm':'국가과학기술표준분류명',
                        'kor_kywd':'한글키워드', 'eng_kywd':'영문키워드', 
                        'rsch_goal_abstract':'연구목표요약', 'rsch_abstract':'연구내용요약', 'exp_efct_abstract':'기대효과요약', 
                        'pjt_prfrm_org_cd':'과제수행기관코드', 'pjt_prfrm_org_nm':'과제수행기관명',
                        'kor_pjt_nm':'국문과제명', 'eng_pjt_nm':'영문과제명', 
                        'pjt_mgnt_org_cd':'과제관리전문기관코드', 'spclty_org_nm':'과제관리전문기관명', 
                        'prctuse_nm':'실용화대상여부', 'rnd_phase':'연구개발단계', 'rsch_exec_suj':'연구수행주체', 'dtl_pjt_clas':'연구개발과제성격',
                        'tech_lifecyc_nm':'기술수명주기', 'regn_nm':'지역', 'pjt_no':'(기관)세부과제번호',
                        'tot_rsch_start_dt':'총연구기간시작일자', 'tot_rsch_end_dt':'총연구기간종료일자', 
                        'tsyr_rsch_start_dt':'당해년도연구시작일자','tsyr_rsch_end_dt':'당해년도연구종료일자',
                        'rndco_tot_amt':'연구비합계금액',
                    }
                },
            ],
            "analysis_data": {
                "분석명": "분석결과"
            }
        ]
    }
    ```
    Analysis List
    ---
    지원 분석 목록
    - lda: 토픽모델링 /analysis/lda
    - sna: (문서/용어) 네트워크분석 /analysis/sna
    - rec_kwd: 단어추천 목록  /analysis/rec_kwd
    - wordcloud: 워드클라우드 /analysis/wordcloud

    Example
    ---
    ```python
    {
        "target_analysis_list": ["sna","wordcloud","rec_kwd","lda"],
        "doc_size": 10,
        "include_doc_data" : true,
        "search_form": {
            "search_text": ["인공지능 빅데이터"],
            "ordered_text": ["인공지능","빅데이터"],
            "exclude_text": ["조류독감"],
            "filter": [{"doc_section":["LA","UK"]}]
        }
    }
    ```
    """
    # 키워드 검색
    doc_size = request.data.get('doc_size',None)
    search_form = request.data.get('search_form',None)
    target_analysis_list = request.data.get('target_analysis_list', [])
    include_doc_data = request.data.get('include_doc_data',True)
    filter_percent = request.data.get('filter_percent', 10)
    use_cache = request.data.get('use_cache',True)

    start_time = timeit.default_timer()
    # validation
    # if doc_size > 1001:
    #     return Response(status=APPLY_400_INVALID_DOC_SIZE.status, data={'error':APPLY_400_INVALID_DOC_SIZE.message})
    if not search_form:
        return Response(status=APPLY_400_SEARCH_FORM_REQUIRED.status, data={'error': APPLY_400_SEARCH_FORM_REQUIRED.message})
    for target_analysis in target_analysis_list:
        if not target_analysis or not target_analysis in SUPPORT_ANALYSIS_TYPES:
            return Response(status=APPLY_400_INVALID_ANALYSIS_TYPE.status, data={'error':APPLY_400_INVALID_ANALYSIS_TYPE.message})            
    if not isinstance(use_cache, bool):
        return Response(status=status.HTTP_400_BAD_REQUEST, data={'error': 'use_cache must be boolean value'})
    try:
        query = create_keyword_search_query(search_form=search_form, use_simple_query=False)
        # search_result = es_query.get_docs_by_full_query(query=query, doc_size=doc_size)
        search_result = get_search_result_by_full_query(query=query, filter_percent=filter_percent)
        if 'lda' in target_analysis_list:
            if len(search_form['search_text'])>0:
                response_format = get_analysis_response(search_result=search_result, target_analysis_list=target_analysis_list, include_doc_data=include_doc_data, search_form=search_form, required_doc_size=doc_size, use_cache=use_cache)
            else:
                return Response(status=APPLY_400_SEARCH_FORM_REQUIRED.status, data={'error':'lda analysis required search_text field value'})
        else:
            response_format = get_analysis_response(search_result=search_result, target_analysis_list=target_analysis_list, include_doc_data=include_doc_data, required_doc_size=doc_size, use_cache=use_cache)
        parsed_full_query = json.dumps(query, ensure_ascii=False)
        response_format.update({'full_query':parsed_full_query})
        return Response(status=status.HTTP_200_OK, data=response_format)
    except RequestError as e: 
        return Response(status=status.HTTP_400_BAD_REQUEST, data={'error': str(e)})
    except TypeError as te:
        return Response(status=status.HTTP_400_BAD_REQUEST, data={'error': str(te)})
    except FailedQueryParsing as fqp:
        return Response(status=status.HTTP_500_INTERNAL_SERVER_ERROR, data={'error': str(fqp)})
    except TooManyClause as tmc:
            return Response(status=status.HTTP_400_BAD_REQUEST, data={'error': str(tmc)})
    except Exception as e:
        return Response(status=status.HTTP_500_INTERNAL_SERVER_ERROR, data={'error': str(e)})
    finally:
        if settings.DEBUG:
            print('analysis_target:',target_analysis_list,'took',timeit.default_timer()-start_time)

body_query_string = openapi.Schema(
    type=openapi.TYPE_OBJECT,
    required=['search_form'],
    example=create_api_format_for_anlaysis(target=['search_form'])['example'],
    properties=create_api_format_for_anlaysis(target=['search_form'],doc_size=300)['properties'],
)
@swagger_auto_schema(
    method='post', 
    request_body=body_query_string,
    responses={
    200 : '과제 별 검색 결과 데이터',
    400 : 
        APPLY_400_SEARCH_FORM_REQUIRED.as_md() +
        APPLY_400_INVALID_DOC_SIZE.as_md() +
        APPLY_400_INVALID_ANALYSIS_TYPE.as_md(),
    }
)
@api_view(['POST'])
def get_docs_by_subject_list(request):
    """
    과제 문서 목록을 통한 유사 과제 검색
    ---
    Request
    ---
    ```python
    {   
        "target_analysis_list": type: list(str) | required | comment: ["분석모듈명1","분석모듈명2"],
        "doc_size": type: int | default: None | comment:조회/분석할 문서 수 제한 (deprecated, 빠른 응답을 위한 테스트 용도로만 사용할 것),
        "include_doc_data" : type:boolean | default: true | comment:반환값에 문서데이터 첨부여부. 분석결과만 필요 시 false,
        "use_cache": boolean | default: true | comment: 미리 분석된 캐시데이터가 있다면 사용
        "search_form" : [
            {
                "id": "",
                "prog_nm": "",
                "target_subject_list": [
                    {
                        "kor_pjt_nm":"과제제목",
                        "rsch_goal_abstract":"연구목표",
                        "rsch_abstract":"연구내용",
                        "exp_efct_abstract":"기대효과",
                        "kor_kywd":["한글키워드1",],
                        "eng_kywd":["영문키워드1",],
                        "rndco_tot_amt": 연구비합계금액
                    }
                ]
            }
        ]
    }
    ```
    Response
    ----
    ```Python
    "search_form" : [
        {
            "id": "",
            "prog_nm": "",
            "target_subject_list": [
                {
                    "kor_pjt_nm":"과제제목",
                    "rsch_goal_abstract":"연구목표",
                    "rsch_abstract":"연구내용",
                    "exp_efct_abstract":"기대효과",
                    "kor_kywd":["한글키워드1",],
                    "eng_kywd":["영문키워드1",],
                    "rndco_tot_amt": 연구비합계금액
                    "search_result": {
                        "full_query": "검색에 사용된 쿼리, 시각화 요청 시 인자값으로 전달",
                        "data": [
                            "doc_ids": ["문서id1",],
                            "total": "검색된 전체 문서수",
                            "doc_size": "제공 문서수",
                            "doc_data": [
                                {
                                    "_index": target_db 인덱스명,
                                    "_type": 엘라스틱서치 인덱스 타입 (_doc으로 고정되어 있음),
                                    "_id": "문서id",
                                    "_score": "검색 결과 스코어",
                                    "_rltd":"키워드 관련성",
                                    "_infl":"과제 영향력",
                                    "_btwn":"과제간 매개력",
                                    "_degr":"과제 연결성",
                                    "_rcnt":"과제 연도"
                                    "_routing": "문서 소속 shard 내 routing 경로",
                                    "_source": {
                                        'doc_id':'문서id(과제고유번호와 동일)', 
                                        'pjt_id':'과제고유번호', 'stan_yr':'기준년도', yr_cnt: '연차','hm_nm':'연구책임자명',
                                        'rsch_area_cls_cd':'국가과학기술표준분류코드','rsch_area_cls_nm':'국가과학기술표준분류명',
                                        'kor_kywd':'한글키워드', 'eng_kywd':'영문키워드', 
                                        'rsch_goal_abstract':'연구목표요약', 'rsch_abstract':'연구내용요약', 'exp_efct_abstract':'기대효과요약', 
                                        'pjt_prfrm_org_cd':'과제수행기관코드', 'pjt_prfrm_org_nm':'과제수행기관명',
                                        'kor_pjt_nm':'국문과제명', 'eng_pjt_nm':'영문과제명', 
                                        'pjt_mgnt_org_cd':'과제관리전문기관코드', 'spclty_org_nm':'과제관리전문기관명', 
                                        'prctuse_nm':'실용화대상여부', 'rnd_phase':'연구개발단계', 'rsch_exec_suj':'연구수행주체', 'dtl_pjt_clas':'연구개발과제성격',
                                        'tech_lifecyc_nm':'기술수명주기', 'regn_nm':'지역', 'pjt_no':'(기관)세부과제번호',
                                        'tot_rsch_start_dt':'총연구기간시작일자', 'tot_rsch_end_dt':'총연구기간종료일자', 
                                        'tsyr_rsch_start_dt':'당해년도연구시작일자','tsyr_rsch_end_dt':'당해년도연구종료일자',
                                        'rndco_tot_amt':'연구비합계금액',
                                    }
                                },
                            ],
                            "analysis_data": {
                                "분석명": "분석결과"
                            }
                        ]
                    }
                }
            ]
        },
    ]
    ```
    Analysis List
    ---
    지원 분석 목록
    - sna: (문서/용어) 네트워크분석 /analysis/sna
    - rec_kwd: 단어추천 목록  /analysis/rec_kwd
    - wordcloud: 워드클라우드 /analysis/wordcloud    

    Example
    ---
    ```
    {
        "target_analysis_list":["sna","wordcloud","rec_kwd"],
        "include_doc_data" : false,
        "search_form" : [
            {
                "id": "1",
                "prog_nm": "사업명1",
                "target_subject_list": [
                    {
                        "kor_pjt_nm":"과제제목",
                        "rsch_goal_abstract":"연구목표",
                        "rsch_abstract":"연구내용",
                        "exp_efct_abstract":"기대효과",
                        "kor_kywd":["한글키워드1"],
                        "eng_kywd":["영문키워드1"],
                        "rndco_tot_amt": 1000000
                    }
                ]
            }
        ]
    }
    ```
    """
    # 과제문서 검색
    doc_size = request.data.get('doc_size',None)
    search_form = request.data.get('search_form',None)
    target_analysis_list = request.data.get('target_analysis_list', [])
    include_doc_data = request.data.get('include_doc_data',True)
    filter_percent = request.data.get('filter_percent', 10)
    use_cache = request.data.get('use_cache',True)
    lda_keyword = request.data.get('lda_keyword',None)

    start_time = timeit.default_timer()

    # validation
    # if doc_size > 1001:
    #     return Response(status=APPLY_400_INVALID_DOC_SIZE.status, data={'error':APPLY_400_INVALID_DOC_SIZE.message})
    
    if not search_form or len(search_form)==0 :
        return Response(status=APPLY_400_SEARCH_FORM_REQUIRED.status, data={'error': APPLY_400_SEARCH_FORM_REQUIRED.message})
    for target_analysis in target_analysis_list:
        if not target_analysis or not target_analysis in SUPPORT_ANALYSIS_TYPES:
            return Response(status=APPLY_400_INVALID_ANALYSIS_TYPE.status, data={'error':APPLY_400_INVALID_ANALYSIS_TYPE.message})
        if target_analysis == 'lda' and lda_keyword is None:
            return Response(status=status.HTTP_400_BAD_REQUEST, data={'error':'lda_keyword is required for lda analysis'})
    try:
        for _form in search_form:
            if not 'target_subject_list' in _form:
                return Response(status=status.HTTP_400_BAD_REQUEST, data={'error': 'target_subject_list key in search form is required '})
            for target_subject in _form['target_subject_list']:
                query = create_subject_search_query(target_subject=target_subject, use_simple_query=False)
                search_result = get_search_result_by_full_query(query=query, filter_percent=filter_percent)
                response_format = get_analysis_response(search_result=search_result, target_analysis_list=target_analysis_list, include_doc_data=include_doc_data, required_doc_size=doc_size, use_cache=use_cache)
                parsed_query = json.dumps(query, ensure_ascii=False)
                response_format.update({'full_query':parsed_query})
                target_subject.update({'search_result':response_format})
        return Response(status=status.HTTP_200_OK, data=search_form)
    except RequestError as e: 
        return Response(status=status.HTTP_400_BAD_REQUEST, data={'error': str(e)})
    except TooManyClause as tmc:
            return Response(status=status.HTTP_400_BAD_REQUEST, data={'error': str(tmc)})   
    except FailedQueryParsing as fqp:
        return Response(status=status.HTTP_500_INTERNAL_SERVER_ERROR, data={'error': str(fqp)})
    finally:
        if settings.DEBUG:
            print('analysis_target:',target_analysis_list, 'took',timeit.default_timer()-start_time)

body_query_string = openapi.Schema(
    type=openapi.TYPE_OBJECT,
    required=['doc_ids'],
    example=create_api_format_for_anlaysis(target=['search_form'])['example'],
    properties=create_api_format_for_anlaysis(target=['search_form'],doc_size=300)['properties'],
)
@swagger_auto_schema(
    method='post', 
    request_body=body_query_string,
    responses={
    200 : '키워드 검색 결과 데이터',
    400 : 
        APPLY_400_SEARCH_FORM_REQUIRED.as_md() +
        APPLY_400_INVALID_DOC_SIZE.as_md() +
        APPLY_400_INVALID_ANALYSIS_TYPE.as_md(),
    }
)
@api_view(['POST'])
def get_docs_by_doc_ids(request):
    """
    pk 를 통한 문서 검색
    ---
    Request
    ---
    ```python
    {
        "target_analysis_list": type: list(str) | required | comment: ["분석모듈명1","분석모듈명2"],
        "include_doc_data" : type:boolean | default: true | comment:반환값에 문서데이터 첨부여부. 분석결과만 필요 시 false,
        "use_cache": boolean | default: true | comment: 미리 분석된 캐시데이터가 있다면 사용
        "doc_ids" : type:list(str) | required | comment: ["문서아이디1"]
    }
    ```
    Response
    ---
    ```python
    {
        "data": [
            "doc_ids": ["문서id1",],
            "total": "검색된 전체 문서수",
            "doc_size": "제공 문서수",
            "doc_data": [
                {
                    "_index": target_db 인덱스명,
                    "_type": 엘라스틱서치 인덱스 타입 (_doc으로 고정되어 있음),
                    "_id": "문서id",
                    "_score": "검색 결과 스코어",
                    "_rltd":"키워드 관련성",
                    "_infl":"과제 영향력",
                    "_btwn":"과제간 매개력",
                    "_degr":"과제 연결성",
                    "_rcnt":"과제 연도"
                    "_routing": "문서 소속 shard 내 routing 경로",
                    "_source": {
                        'doc_id':'문서id(과제고유번호와 동일)', 
                        'pjt_id':'과제고유번호', 'stan_yr':'기준년도', yr_cnt: '연차','hm_nm':'연구책임자명',
                        'rsch_area_cls_cd':'국가과학기술표준분류코드','rsch_area_cls_nm':'국가과학기술표준분류명',
                        'kor_kywd':'한글키워드', 'eng_kywd':'영문키워드', 
                        'rsch_goal_abstract':'연구목표요약', 'rsch_abstract':'연구내용요약', 'exp_efct_abstract':'기대효과요약', 
                        'pjt_prfrm_org_cd':'과제수행기관코드', 'pjt_prfrm_org_nm':'과제수행기관명',
                        'kor_pjt_nm':'국문과제명', 'eng_pjt_nm':'영문과제명', 
                        'pjt_mgnt_org_cd':'과제관리전문기관코드', 'spclty_org_nm':'과제관리전문기관명', 
                        'prctuse_nm':'실용화대상여부', 'rnd_phase':'연구개발단계', 'rsch_exec_suj':'연구수행주체', 'dtl_pjt_clas':'연구개발과제성격',
                        'tech_lifecyc_nm':'기술수명주기', 'regn_nm':'지역', 'pjt_no':'(기관)세부과제번호',
                        'tot_rsch_start_dt':'총연구기간시작일자', 'tot_rsch_end_dt':'총연구기간종료일자', 
                        'tsyr_rsch_start_dt':'당해년도연구시작일자','tsyr_rsch_end_dt':'당해년도연구종료일자',
                    }
                },
            ],
            "analysis_data": {
                "분석명": "분석결과"
            }
        ]
    }
    ```
    Analysis List
    ---
    지원 분석 목록
    - sna: (문서/용어) 네트워크분석 /analysis/sna
    - rec_kwd: 단어추천 목록  /analysis/rec_kwd
    - wordcloud: 워드클라우드 /analysis/wordcloud
    
    Example
    ---
    ```python
    {
        "target_analysis_list": ["sna","wordcloud","rec_kwd"],
        "include_doc_data" : true,
        "doc_ids": ["1315000984","1325163840"]
    }
    ```
    """
    # 문서아이디
    target_db = TARGET_DB
    doc_ids = request.data.get('doc_ids',None)
    target_analysis_list = request.data.get('target_analysis_list', [])
    include_doc_data = request.data.get('include_doc_data',True)
    use_cache = request.data.get('use_cache',True)

    start_time = timeit.default_timer()
    # validation
    if not doc_ids:
        return Response(status=APPLY_400_DOC_IDS_REQUIRED.status, data={'error':APPLY_400_DOC_IDS_REQUIRED.message})
    for target_analysis in target_analysis_list:
        if not target_analysis or not target_analysis in SUPPORT_ANALYSIS_TYPES or target_analysis == 'lda':
            return Response(status=APPLY_400_INVALID_ANALYSIS_TYPE.status, data={'error':APPLY_400_INVALID_ANALYSIS_TYPE.message})        
    # doc_ids = doc_ids.split(',')
    # if doc_size > 1001:
    #     return Response(status=APPLY_400_INVALID_DOC_SIZE.status, data={'error':APPLY_400_INVALID_DOC_SIZE.message})
    es_query = ElasticQuery(target_db=target_db)
    try:
        # 분석 종류에 따라 doc_ids를 사용할 지 어떨지 확인 필요
        search_result = es_query.get_docs_by_doc_ids(doc_ids=doc_ids, source=ES_DOC_SOURCE)
        response_format = get_analysis_response(search_result=search_result, target_analysis_list=target_analysis_list, include_doc_data=include_doc_data, use_cache=use_cache)
        return Response(status=status.HTTP_200_OK, data=response_format)
    except RequestError as e: 
        return Response(status=status.HTTP_400_BAD_REQUEST, data={'error': str(e)})
    finally:
            if settings.DEBUG:
                print('analysis_target:',target_analysis_list,'doc_size:',len(search_result['hits']['hits']),'took',timeit.default_timer()-start_time)

body_query_string = openapi.Schema(
    type=openapi.TYPE_OBJECT,
    required=['full_query'],
    example=create_api_format_for_anlaysis(target=['search_form'])['example'],
    properties=create_api_format_for_anlaysis(target=['search_form'],doc_size=300)['properties'],
)
@swagger_auto_schema(
    method='post', 
    request_body=body_query_string,
    responses={
    200 : '키워드 검색 결과 데이터',
    400 : 
        APPLY_400_SEARCH_FORM_REQUIRED.as_md() +
        APPLY_400_INVALID_DOC_SIZE.as_md() +
        APPLY_400_INVALID_ANALYSIS_TYPE.as_md(),
    }
)
@api_view(['POST'])
def analyze_docs_by_full_query(request):
    """
    풀쿼리 문서 검색 및 분석 결과 반환
    ---
    Request
    ---
    ```python
    {
        "full_query" : type:str | required | comment: 분석엔진에서 제공한 검색식. 시각화 대상 문서 추출에 사용함.
        "include_doc_data" : type:boolean | default: true | comment:반환값에 문서데이터 첨부여부. 분석결과만 필요 시 false,
        "doc_size": type: int | default: None | comment:조회/분석할 문서 수 제한 (deprecated, 빠른 응답을 위한 테스트 용도로만 사용할 것),
        "doc_ids": type: list(str) | default: [] | comment: 특정 문서id 리스트에 한정해서 분석을 진행할 경우 요청,
        "use_cache": boolean | default: true | comment: 미리 분석된 캐시데이터가 있다면 사용
        "target_analysis_list": type: list(str) | default: ['sna','wordcloud'] (추후 추가 예정) | comment: 특정 분석만 필요할 경우 요청. 그 외 기본값으로 필요 분석 모두 수행.
    }
    ```
    Response
    ---
    ```python
    {
        "data": [
            "doc_ids": ["문서id1",],
            "total": "검색된 전체 문서수",
            "doc_size": "제공 문서수",
            "doc_data": [
                {
                    "_index": target_db 인덱스명,
                    "_type": 엘라스틱서치 인덱스 타입 (_doc으로 고정되어 있음),
                    "_id": "문서id",
                    "_score": "검색 결과 스코어",
                    "_rltd":"키워드 관련성",
                    "_infl":"과제 영향력",
                    "_btwn":"과제간 매개력",
                    "_degr":"과제 연결성",
                    "_rcnt":"과제 연도"
                    "_routing": "문서 소속 shard 내 routing 경로",
                    "_source": {
                        'doc_id':'문서id(과제고유번호와 동일)', 
                        'pjt_id':'과제고유번호', 'stan_yr':'기준년도', yr_cnt: '연차','hm_nm':'연구책임자명',
                        'rsch_area_cls_cd':'국가과학기술표준분류코드','rsch_area_cls_nm':'국가과학기술표준분류명',
                        'kor_kywd':'한글키워드', 'eng_kywd':'영문키워드', 
                        'rsch_goal_abstract':'연구목표요약', 'rsch_abstract':'연구내용요약', 'exp_efct_abstract':'기대효과요약', 
                        'pjt_prfrm_org_cd':'과제수행기관코드', 'pjt_prfrm_org_nm':'과제수행기관명',
                        'kor_pjt_nm':'국문과제명', 'eng_pjt_nm':'영문과제명', 
                        'pjt_mgnt_org_cd':'과제관리전문기관코드', 'spclty_org_nm':'과제관리전문기관명', 
                        'prctuse_nm':'실용화대상여부', 'rnd_phase':'연구개발단계', 'rsch_exec_suj':'연구수행주체', 'dtl_pjt_clas':'연구개발과제성격',
                        'tech_lifecyc_nm':'기술수명주기', 'regn_nm':'지역', 'pjt_no':'(기관)세부과제번호',
                        'tot_rsch_start_dt':'총연구기간시작일자', 'tot_rsch_end_dt':'총연구기간종료일자', 
                        'tsyr_rsch_start_dt':'당해년도연구시작일자','tsyr_rsch_end_dt':'당해년도연구종료일자',
                        'rndco_tot_amt':'연구비합계금액',
                    }
                },
            ],
            "analysis_data": {
                "분석명": "분석결과"
            }
        ]
    }
    ```
    Analysis List
    ---
    기본 제공 분석 목록
    - sna: (문서/용어) 네트워크분석 /analysis/sna
    - rec_kwd: 단어추천 목록  /analysis/rec_kwd
    - player_sna: 플레이어 네트워크 분석 /analysis/player_sna
    - wordcloud: 워드클라우드 /analysis/wordcloud
    - bubble_chart: 버블차트 /analysis/bubble_chart
    - bar_chart: 바차트 /analysis/bar_chart
    - line_chart: 라인차트 /analysis/line_chart
    - chart_source: 문서 관련 데이터 /analysis/chart_source

    Example
    ---
    ```python
     {
        
        "full_query": {
            "_source": [
                "*"
            ],
            "query": {
                "bool": {
                    "must": [
                        {
                            "query_string": {
                                "fields": [
                                    "kor_kywd^1.5",
                                    "eng_kywd^1.5",
                                    "kor_pjt_nm",
                                    "rsch_goal_abstract",
                                    "rsch_abstract",
                                    "exp_efct_abstract"
                                ],
                                "query": "(인공지능 빅데이터|\\"인공지능 빅데이터\\"^1.5|인공지능^2.0|빅데이터^1.5) + -(조류독감)"
                            }
                        }
                    ],
                    "filter": [
                        {
                            "terms": {
                                "doc_section": [
                                    "LA",
                                    "UK"
                                ]
                            }
                        }
                    ]
                }
            },
            "min_score": 23
        },
        "include_doc_data": true
    }
    ```
    """
    print('\033[95m' + "=========================================================" + '\033[0m')
    print('\033[95m' + " get chart data start >> " + str(timezone.now()) + '\033[0m')
    print('\033[95m' + "=========================================================" + '\033[0m')
    # 풀쿼리 검색
    full_query = request.data.get('full_query',None)
    doc_ids = request.data.get('doc_ids',None)
    target_analysis_list = request.data.get('target_analysis_list', ["sna","player_sna","wordcloud","bubble_chart","bar_chart","line_chart","chart_source"])
    include_doc_data = request.data.get('include_doc_data',True)
    filter_percent = request.data.get('filter_percent', 10)
    doc_size = request.data.get('doc_size',None)
    use_cache = request.data.get('use_cache',True)

    start_time = timeit.default_timer()
    # validation
    if not full_query:
        return Response(status=APPLY_400_FULL_QUERY_REQUIRED.status, data={'error':APPLY_400_FULL_QUERY_REQUIRED.message})
    
    if isinstance(full_query,str):
        try:
            full_query = json.loads(full_query)
        except json.decoder.JSONDecodeError:
            return Response(status=status.HTTP_400_BAD_REQUEST, data={'error': 'malformed query format'})  
    try:
        es_query = ElasticQuery(target_db=TARGET_DB)
        # 문서를 선택한 상태에서 시각화 요청한 케이스를 위한 adhoc
        # 선택된 문서들은 시각화 문서 집단내 모두 포함되어야함 => 각 분석 단계에서 score 기준으로 문서 필터링이 이루어짐 => doc_ids 로 검색된 문서들은 max_score 할당해서 분석과정에서 살아 남을 수 있게 처리
        search_result = get_search_result_by_full_query(query=full_query, filter_percent=filter_percent)
        if doc_ids is not None and len(doc_ids) > 0:
            max_score = search_result['hits']['max_score']
            if max_score is None:
                max_score = 1
            print('\033[95m' + "=========================================================" + '\033[0m')
            print('\033[95m' + " elastic search start >> " + str(timezone.now()) + '\033[0m')
            print('\033[95m' + "=========================================================" + '\033[0m')
            doc_ids_search_result = es_query.get_docs_by_doc_ids(doc_ids=doc_ids, source=ES_DOC_SOURCE)
            print('\033[95m' + "=========================================================" + '\033[0m')
            print('\033[95m' + " elastic search end >> " + str(timezone.now()) + '\033[0m')
            print('\033[95m' + "=========================================================" + '\033[0m')
            doc_ids_df = pd.DataFrame(doc_ids_search_result['hits']['hits'])
            search_df =  pd.DataFrame(search_result['hits']['hits'])
            doc_ids_df['_score'] = max_score
            combined_df = pd.concat([search_df,doc_ids_df],axis=0, ignore_index=True)
            combined_df = combined_df.sort_values(['_score'],ascending=False).drop_duplicates('_id').reset_index(drop=True)
            combined_df = combined_df.fillna('')
            search_result['took'] = search_result['took'] + doc_ids_search_result['took']
            search_result['hits']['total']['value'] = search_result['hits']['total']['value'] + doc_ids_search_result['hits']['total']['value']
            search_result['hits']['max_score'] = max_score
            search_result['hits']['hits'] = combined_df.to_dict('records')
        print('\033[95m' + "=========================================================" + '\033[0m')
        print('\033[95m' + " get analysis response start >> " + str(timezone.now()) + '\033[0m')
        print('\033[95m' + "=========================================================" + '\033[0m')
        response_format = get_analysis_response(search_result=search_result, target_analysis_list=target_analysis_list, include_doc_data=include_doc_data, required_doc_size=doc_size, use_cache=use_cache)
        print('\033[95m' + "=========================================================" + '\033[0m')
        print('\033[95m' + " get analysis response end >> " + str(timezone.now()) + '\033[0m')
        print('\033[95m' + "=========================================================" + '\033[0m')
        parsed_full_query = json.dumps(full_query, ensure_ascii=False)
        response_format.update({'full_query':parsed_full_query})
        # response_format['data']['analysis_data'].update(sna_result['data']['analysis_data'])
        print('\033[95m' + "=========================================================" + '\033[0m')
        print('\033[95m' + " get chart data end >> " + str(timezone.now()) + '\033[0m')
        print('\033[95m' + "=========================================================" + '\033[0m')
        return Response(status=status.HTTP_200_OK, data=response_format)
    except RequestError as e: 
        return Response(status=status.HTTP_400_BAD_REQUEST, data={'error': str(e)})    
    except FailedQueryParsing as fqp:
        return Response(status=status.HTTP_500_INTERNAL_SERVER_ERROR, data={'error': str(fqp)})
    except TooManyClause as tmc:
        return Response(status=status.HTTP_400_BAD_REQUEST, data={'error': str(tmc)})        
    finally:
            if settings.DEBUG:
                print('analysis_target:',target_analysis_list,'took',timeit.default_timer()-start_time)

@api_view(['GET'])
def get_category(request):
    response_format = {'data':{'section':{},'class':{},'subclass':{}}}
    required_field = ['section_cd','section_nm','class_cd','class_nm','subclass_cd','subclass_nm']
    category_obj = Category.objects.using(TARGET_DB).all()
    cat_df = pd.DataFrame(category_obj.values())
    try:
        cat_df = cat_df[required_field]
        section_dict = cat_df[['section_cd','section_nm']].drop_duplicates().set_index('section_cd').to_dict()
        class_dict = cat_df[['class_cd','class_nm']].drop_duplicates().set_index('class_cd').to_dict()
        subclass_dict = cat_df[['subclass_cd','subclass_nm']].drop_duplicates().set_index('subclass_cd').to_dict()
        response_format['data']['section'].update(section_dict['section_nm'])
        response_format['data']['class'].update(class_dict['class_nm'])
        response_format['data']['subclass'].update(subclass_dict['subclass_nm'])
        return Response(status=status.HTTP_200_OK, data=response_format)
    except KeyError:
        return Response(status=status.HTTP_500_INTERNAL_SERVER_ERROR, data={'error': 'category data not ready'})

    
