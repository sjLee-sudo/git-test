from numpy import source
import pandas as pd
from rest_framework import status
from elasticsearch.exceptions import RequestError
from utils.custom_errors import APPLY_400_DOC_IDS_REQUIRED, APPLY_400_INVALID_TARGET_DB, APPLY_400_SEARCH_TEXT_REQUIRED, APPLY_400_TARGET_DBS_REQUIRED, APPLY_400_INVALID_ANALYSIS_TYPE, APPLY_400_INVALID_CHART_TYPE, APPLY_400_CHART_TYPE_REQUIRED,APPLY_400_INVALID_QUERY_STRING

from rest_framework.response import Response
from rest_framework.decorators import api_view
from config.constants import DOCUMENT_DB_NAMES, SUPPORT_ANALYSIS_TYPES, CHART_FUNCTION_MAPPING, CHART_MODULE_PATH, TARGET_DB

from analysis.sna.tree import tree_sna_main
from analysis.word_extractors.compound_extractor import EdgelistCompoundExtractor
from analysis.word_extractors.noun_extractor import create_noun_compound_df_from_text_list
from analysis.sna.sna import extract_topic_word
from analysis.edgelist.kistep_edgelist import SnaEdgelist, WordCountEdgelist
from analysis.sna.lda import lda_fullprocess
from analysis.wordcount.d3_chartdata import make_cloud_chart

from db_manager.managers import ElasticQuery

from documents.document_utils import create_keyword_search_query

from drf_yasg import openapi 
from drf_yasg.utils import swagger_auto_schema

import importlib

# dynamic chart function import 
chart_module = importlib.import_module(CHART_MODULE_PATH)

CHART_FUNCTION_SELECTOR = {
}

for chart_key in CHART_FUNCTION_MAPPING.keys():
    CHART_FUNCTION_SELECTOR.update({chart_key: getattr(chart_module,CHART_FUNCTION_MAPPING[chart_key])})

def create_api_format_for_anlaysis(target=[], doc_size=100):    
    examples = {
        'doc_ids': ['문서번호1','문서번호2'],
        'search_text':'검색문장 OR 검색식',
        'target_db': ", ".join(DOCUMENT_DB_NAMES)+' 중 택1',
        'target_dbs':  ",".join(DOCUMENT_DB_NAMES),
        'analysis_type':  ",".join(SUPPORT_ANALYSIS_TYPES) + ' 중 택1',
        'doc_size': '100',
        'term_size': '100',
        'chart_types': ",".join(CHART_FUNCTION_MAPPING.keys()),
        'word_limit_count': '100',
        'is_query_string': "true, false 중 택 1",
        'check_exists_only' : 'true, false 중 택 1, 기본값 false',
        'query_string':'(딥러닝* OR 머신러닝* OR 인공지능* OR 조류독감* OR 신종플루*) AND (딥러닝 OR 딥러닝 OR deep learning)',
        'search_form':"{'key':'value'}"
    }
    properties = {
        'doc_ids': openapi.Schema(
            type=openapi.TYPE_OBJECT,
            description='문서고유번호 리스트'
        ),
        'search_text': openapi.Schema(
            type=openapi.TYPE_STRING,
            description='검색문장 또는 검색식'
        ),
        'target_db': openapi.Schema(
            type=openapi.TYPE_STRING,
            description='분석 대상 데이터베이스명'
        ),
        'target_dbs':openapi.Schema(
            type=openapi.TYPE_STRING,
            description='콤마(,) Separated 된 복수개의 데이터베이스명',
        ),
        'analysis_type':  openapi.Schema(
            type=openapi.TYPE_STRING,
            description='분석 종류'
        ),
        'doc_size': openapi.Schema(
            type=openapi.TYPE_STRING,
            description=f'document 수, 기본값 {doc_size}'
        ),
        'chart_types': openapi.Schema(
            type=openapi.TYPE_STRING,
            description='콤마(,) Separated 된 복수개의 차트 종류'
        ),
        'word_limit_count':openapi.Schema(
            type=openapi.TYPE_STRING,
            description='차트에 나타낼 단어의 개수 제한, 기본값 100 (높은 빈도수 순으로 정렬)',
        ),
        'is_query_string': openapi.Schema(
            type=openapi.TYPE_BOOLEAN,
            description='검색문장이 일반 문장인지 검색식인지 구분. \n true 면 검색식으로 문서조회 \n false면 검색문장으로 문서조회)',
        ),
        'check_exists_only':openapi.Schema(
            type=openapi.TYPE_BOOLEAN,
            description='자료 존재 여부만 확인. true 선택 시 boolean값 반환. false 선택 시 자료 자체 반환 (자료없으면 빈 데이터 {} 반환)'
        ),
        'term_size': openapi.Schema(
            type=openapi.TYPE_STRING,
            description=f'term 수, 기본값 100'
        ),
        'query_string': openapi.Schema(
            type=openapi.TYPE_STRING,
            description='검색식'
        ),
        'search_form': openapi.Schema(
            type=openapi.TYPE_OBJECT,
            description='검색 조건'
        )
    }
    result = {'example':{},'properties':{}}
    for x in target:
        result['example'].update({x:examples[x]})
        result['properties'].update({x:properties[x]})
    return result

body_analysis_cache = openapi.Schema(
    type=openapi.TYPE_OBJECT,
    required=['search_text','target_db','analysis_type'],
    example=create_api_format_for_anlaysis(target=['search_text','target_db','analysis_type','doc_size','check_exists_only'],doc_size=100)['example'],
    properties=create_api_format_for_anlaysis(target=['search_text','target_db','analysis_type','doc_size','check_exists_only'],doc_size=100)['properties']
)

body_analysis_sna = openapi.Schema(
    type=openapi.TYPE_OBJECT,
    required=['search_text','target_db'],
    example=create_api_format_for_anlaysis(target=['search_text','target_db','doc_size','is_query_string'],doc_size=100)['example'],
    properties=create_api_format_for_anlaysis(target=['search_text','target_db','doc_size','is_query_string'],doc_size=100)['properties']
)
body_analysis_tree_sna =  openapi.Schema(
    type=openapi.TYPE_OBJECT,
    required=['search_text','target_dbs'],
    example=create_api_format_for_anlaysis(target=['search_text','target_dbs','doc_size','is_query_string'],doc_size=100)['example'],
    properties=create_api_format_for_anlaysis(target=['search_text','target_db','doc_size','is_query_string'],doc_size=100)['properties']
)
body_analysis_wordcount = openapi.Schema(
    type=openapi.TYPE_OBJECT,
    required=['search_text','target_db','chart_types'],
    example=create_api_format_for_anlaysis(target=['search_text','target_db','chart_types','doc_size','word_limit_count','is_query_string'],doc_size=100)['example'],
    properties=create_api_format_for_anlaysis(target=['search_text','target_db','chart_types','doc_size','word_limit_count','is_query_string'],doc_size=100)['properties']
)
body_analysis_lda = openapi.Schema(
    type=openapi.TYPE_OBJECT,
    required=['search_text','target_db'],
    example=create_api_format_for_anlaysis(target=['search_text','target_db','doc_size'],doc_size=500)['example'],
    properties=create_api_format_for_anlaysis(target=['search_text','target_db','doc_size'],doc_size=500)['properties']
)
body_analysis_term_recommend = openapi.Schema(
    type=openapi.TYPE_OBJECT,
    required=['search_text','target_db'],
    example=create_api_format_for_anlaysis(target=['search_text','target_db','doc_size','term_size','is_query_string'],doc_size=100)['example'],
    properties=create_api_format_for_anlaysis(target=['search_text','target_db','doc_size','term_size','is_query_string'],doc_size=100)['properties']
)

def get_cache_search_format(target_db, search_text, is_query_string=False):
    target_db_list = target_db.split(',')
    cache_target_db = ",".join(sorted(target_db_list))    
    if is_query_string:
        return cache_target_db, search_text
    combined_search_text_keyword = set()
    for _target_db in target_db_list:
        es_query = ElasticQuery(target_db=_target_db)
        search_text_keyword = es_query.tokenize_text(search_text)
        combined_search_text_keyword.update(search_text_keyword)
    search_text_keyword = ",".join(sorted(list(combined_search_text_keyword)))
    return cache_target_db, search_text_keyword

# @swagger_auto_schema(method='post', request_body=body_analysis_sna, responses={
#     200 : 'topic_word: 주요단어, word_sna: 용어 네트워크 분석 결과, paper_sna: 문서 네트워크 분석 결과',
#     400 : 
#         APPLY_400_DOC_IDS_REQUIRED.as_md() 
# })
@api_view(['POST'])
def get_sna(request):
    """

    네트워크 분석
    ----
    자료 구조:
    - paper와 word의 구조는 동일

    * nodes : 노드정보 | list(dict)
    id : 노드의 고유번호 | int
    t : 노드의 이름 | chr
    _neighbor : 2mode에서 해당 노드와 이웃하는 노드(단어의 경우 해당 단어가 등장하는 문서) | list
    pyear : 출현연도 | list
    section : 해당 섹션 | list
    s : 소시오그램 노드 크기 | float
    sE : 아이겐벡터 중심성 | float    * word network 에서만 존재
    b : 0 ~ 1로 표준화한 paper의 bm25 score | float    * paper network 에서만 존재
 

    * edges : 엣지정보 | list(dict)
    source : source | chr
    target : target | chr
    force : 강제 연결여부 | chr    * word network 에서만 존재
    v : 엣지 가중치 | float
        - 0 ~ 1로 정규화된 pmi | word network
        - 0 ~ 1로 정규화된 두 paper가 공유하는 단어의 갯수 | paper network

    # example
    ```
    {
        'sna': {
            'word_sna': {
                'nodes': [
                    {
                        'id': 0,
                        't': '0.0',
                        '_neighbor': ['100023', '100242'],
                        'pyear': ['2020','2019'],
                        'section': ['LA'],
                        'sE': 0.2658595716,
                        's': 9
                    },
                    {
                        'id': 1354,
                        't': '희석',
                        '_neighbor': ['100023', '100242'],
                        'pyear': ['2020','2019'],
                        'section': ['LA'],
                        'sE': 0.2658595716
                    }
                ],
                'links': [
                    {'source': '31', 'target': '73', 'v': 0.2851283835, 'force': ''},
                    {'source': '31', 'target': '48', 'v': 0.3094291352, 'force : 'T'},
                ]
            },
            'paper_sna': {
                'nodes': [
                    {
                        'id': 0,
                        'doc_id': '1425145402',
                        'pyear': ['2020'],
                        'section': ['LA'],
                        't': '인공지능(AI) 기반 위암 유전자 서비스',
                        's': 0,
                        'b': 0.1986029097
                    },
                    {
                        'id': 2,
                        'doc_id': '1345324754',
                        'pyear': ['2020'],
                        'section': ['UK'],
                        't': '인공지능(AI) 시대에 사이버보안에 관한 법제연구',
                        's': 16,
                        'b': 0.5467511683                       
                    }
                ],
                'links': [
                    {'source': '18', 'target': '19', 'v': 0.7647058824, 'force': ''},
                    {'source': '13', 'target': '22', 'v': 0.0588235294, 'force : ''},
                ]
            }
        }
    }        
    ```
    """
    target_db = TARGET_DB
    doc_ids = request.data.get('doc_ids',None)
    
    # validation
    if not doc_ids:
        return Response(status=APPLY_400_DOC_IDS_REQUIRED.status, data={'error':APPLY_400_DOC_IDS_REQUIRED.message})
    doc_ids = doc_ids.split(',')
    sorted_doc_ids = ','.join(sorted(doc_ids))
    doc_size = len(doc_ids)
    result_dict = {}
    # cache result
    es_query = ElasticQuery(target_db=target_db)
    cache_data = es_query.get_analysis_cache_by_search_keys(target_db=target_db, search_keys=sorted_doc_ids, analysis_type='sna', doc_size=doc_size)
    if cache_data:
        return Response(status=status.HTTP_200_OK, data=cache_data['analysis_result'])
    # edgelist cache check
    cache_edgelist = es_query.get_analysis_cache_by_search_keys(target_db=target_db, search_keys=sorted_doc_ids, analysis_type='sna_edgelist', doc_size=doc_size)
    if cache_edgelist:
        edge_dict = cache_edgelist['analysis_result']
        ego_edge = pd.DataFrame(edge_dict['ego_edge'])
        refined_ego_edge = pd.DataFrame(edge_dict['refined_ego_edge'])
    else:
        sna_edge = SnaEdgelist()
        ego_edge, refined_ego_edge = sna_edge.get_sna_refined_ego_edgelist(target_db=target_db, doc_ids=doc_ids, size=doc_size)
        es_query.insert_analysis_cache(target_db=target_db, search_keys=sorted_doc_ids, analysis_type='sna_edgelist', doc_size=doc_size, analysis_result={'ego_edge':ego_edge.to_dict(),'refined_ego_edge':refined_ego_edge.to_dict()})

    # sna analysis
    topic_word = '' 
    word_json = ''
    paper_json = ''
    if not ego_edge.empty and not refined_ego_edge.empty:
        topic_word, word_json, paper_json = extract_topic_word(ego_edge, refined_ego_edge)
    result_dict.update({'doc_ids':doc_ids, 'word_sna':word_json, 'paper_sna':paper_json, 'topic_word':topic_word})

    # cache insert
    es_query.insert_analysis_cache(target_db=target_db, search_keys=sorted_doc_ids, analysis_type='sna', doc_size=doc_size, analysis_result=result_dict)

    return Response(status=status.HTTP_200_OK, data=result_dict)

# @swagger_auto_schema(method='post', 
# request_body=body_analysis_tree_sna, responses={
#     200 : '다차원 네트워크 분석 결과',
#     400 : 
#         APPLY_400_INVALID_TARGET_DB.as_md() +
#         APPLY_400_SEARCH_TEXT_REQUIRED.as_md(), 
# })
# @api_view(['POST'])
def get_tree_sna(request):
    """
    다차원 네트워크 분석
    ----
    자료구조:
    * nodes
    id : 노드의 고유번호 | int
    index_key : 해당 노드의 유형 | chr
    name : 노드의 이름 | chr
    depth : 트리 단계 구분 | chr
    link_size : 연결된 엣지의 수 | int

    * edges
    source : source | chr
    target : target | chr
    rel : 연결관계 구분 | chr
    weight : 엣지의 가중치 | int
    section : 해당 엣지리스트의 섹션 : chr
    - 전체 데이터(T)에서 생성된 경우에는 섹션값이 비어있음

    * outer_links
    paper_json 구조와 동일

    * search_text
    search_text : 검색어 | int

    # example
    ```
    {
        'nodes' : [
            {
                'id': 'db_parentSA'
                'index_key': 'section',
                'name' : 'A',
                'depth' : '2-3',
                'link_size' : 14
            }
        ],
        'edges' : [
            {
                'source' : 'db_parentSA',
                'target' : 'db_parentP1',
                'rel' : '3-section-paper',
                'weight' : 1
            }
        ],
        'outer_links' : [
            {
                'source':'db_parentP1', 'target':'db_parentP12', 'v':1
            }
        ],
        'search_text' : '인공지능'
    }
    ```
    """
    search_text = request.data.get('search_text',None)
    is_query_string = request.data.get('is_query_string',False)
    target_dbs = request.data.get('target_dbs',None)
    doc_size = request.data.get('doc_size',300)
    if not search_text:
        return Response(status=APPLY_400_SEARCH_TEXT_REQUIRED.status, data={'error':APPLY_400_SEARCH_TEXT_REQUIRED.message})
    if not target_dbs:
        return Response(status=APPLY_400_TARGET_DBS_REQUIRED.status, data={'error':APPLY_400_TARGET_DBS_REQUIRED.message})

    target_db_list = target_dbs.split(',')
    for _target_db in target_db_list:
        if not _target_db in DOCUMENT_DB_NAMES:
            return Response(status=APPLY_400_INVALID_TARGET_DB.status, data={'error':APPLY_400_INVALID_TARGET_DB.message})
    if isinstance(is_query_string,str):
        is_query_string = is_query_string.upper() == 'TRUE'

    # cache result
    es_query = ElasticQuery(target_db=target_db_list[0])
    
    if is_query_string:
        valid_result = es_query.validate_query_string(query_string=search_text)
        if not valid_result['valid']:
            return Response(status=APPLY_400_INVALID_QUERY_STRING.status, data={'error': valid_result['error']})
    
    cache_target_db, search_text_keyword = get_cache_search_format(target_db=target_dbs, search_text=search_text, is_query_string=is_query_string)
    cache_data = es_query.get_analysis_cache_by_search_keys(target_db=cache_target_db, search_keys=search_text_keyword, analysis_type='tree_sna', doc_size=doc_size)
    if cache_data:
        return Response(status=status.HTTP_200_OK, data=cache_data['analysis_result'])
    
    # tree sna analysis
    tree_sna_result = tree_sna_main(search_text=search_text, target_db_list=target_db_list, is_query_string=is_query_string, size=doc_size)
    
    # cache insert
    es_query.insert_analysis_cache(target_db=cache_target_db, search_keys=search_text_keyword, analysis_type='tree_sna', doc_size=doc_size, analysis_result=tree_sna_result)

    return Response(status=status.HTTP_200_OK, data=tree_sna_result)

# @swagger_auto_schema(method='post', 
# request_body=body_analysis_lda,
# responses={
#     200 : 'topic_model_base: search_text 을 통해  topic model base data',
#     400 : 
#         APPLY_400_INVALID_TARGET_DB.as_md() +
#         APPLY_400_SEARCH_TEXT_REQUIRED.as_md(), 
# })
@api_view(['POST'])
def get_topic_model_base_data(request):
    """
    토픽모델 베이스 데이터
    LDA 기법 활용 토픽 및 토픽 주제어 생성
    ----
    자료 구조:
    * nodes : 노드 정보 | list(dict)
    id : 노드의 고유번호 | int
    name : 노드의 이름 | chr
    type : 노드의 유형(검색어 or 토픽 or 단어) | chr

    * edges : 엣지 정보 | list(dict)
    source : source | int
    target : target | int
    check : 단어 포함 여부 defalut 값(Y/N) | chr
    relation : 노드간 관계 속성 (R: 관련어, 'S':동의어)| str
    * search_text - topic 속성 노드 끼리의 연결에는 check에 None값이 들어감

    # example
    ```
    {
        "lda": {
            'nodes': [
                {
                    'id': 0,
                    'name': '딥러닝',
                    'type' : 'search_text'
                },
                {
                    'id': 1,
                    'name': 'T0',
                    'type' : 'topic'
                },
                {
                    'id' : 2,
                    'name' : '영상기반'
                    'type' : 'word',
                    'weight': 3

                }
            ],
            'links': [
                {'source': 1, 'target': 2, 'check': 'Y', 'relation':'R' },
                {'source': 1, 'target': 3, 'check': 'N', 'relation': 'R' }
            ]
        }
    }
    ```
    """

    search_text = request.data.get('search_text', None)
    target_db = request.data.get('target_db', None)
    doc_size = request.data.get('doc_size',500)
    # validation
    if not search_text:
        return Response(status=APPLY_400_SEARCH_TEXT_REQUIRED.status, data={'error':APPLY_400_SEARCH_TEXT_REQUIRED.message})
    if not target_db in DOCUMENT_DB_NAMES:
        return Response(status=APPLY_400_INVALID_TARGET_DB.status, data={'error':APPLY_400_INVALID_TARGET_DB.message})

    result_dict = {}

    # temp example
    if search_text == '인공지능':
        from analysis.example import TEMP_EXAMPLE
        
        return Response(status=status.HTTP_200_OK, data=TEMP_EXAMPLE)

    # cache result
    es_query = ElasticQuery(target_db=target_db)
    cache_target_db, search_text_keyword = get_cache_search_format(target_db=target_db, search_text=search_text, is_query_string=False)
    cache_data = es_query.get_analysis_cache_by_search_keys(target_db=cache_target_db, search_keys=search_text_keyword, analysis_type='lda', doc_size=doc_size)
    if cache_data:
        return Response(status=status.HTTP_200_OK, data=cache_data['analysis_result'])
    
    # edgelist cache check
    cache_edgelist = es_query.get_analysis_cache_by_search_keys(target_db=cache_target_db, search_keys=search_text_keyword, analysis_type='sna_edgelist', doc_size=doc_size)
    if cache_edgelist:
        edge_dict = cache_edgelist['analysis_result']
        ego_edge = pd.DataFrame(edge_dict['ego_edge'])
        refined_ego_edge = pd.DataFrame(edge_dict['refined_ego_edge'])
    else:
        search_query = create_keyword_search_query(target_db=target_db, search_text=search_text, fields=[])
        search_result = es_query.get_docs_by_full_query(query=search_query,doc_size=doc_size)
        doc_ids = []
        for doc in search_result['hits']['hits']:
            doc_ids.append(doc['_id'])
        sna_edge = SnaEdgelist()
        ego_edge, refined_ego_edge = sna_edge.get_sna_refined_ego_edgelist(target_db=target_db, doc_ids=doc_ids, size=doc_size)
        es_query.insert_analysis_cache(target_db=target_db, search_keys=search_text_keyword, analysis_type='sna_edgelist', doc_size=doc_size, analysis_result={'ego_edge':ego_edge.to_dict(),'refined_ego_edge':refined_ego_edge.to_dict()})
        
    # lda analysis
    lda_result = ''
    if not ego_edge.empty and not refined_ego_edge.empty:
        lda_result = lda_fullprocess(search_text=search_text, _edgelist = refined_ego_edge, remove_n = 0, no_below = 2, no_above = 0.1, _start = 3, _end = 15, p_limit = 0.1)
    result_dict.update(lda_result)

    # cache insert
    es_query.insert_analysis_cache(target_db=target_db, search_keys=search_text_keyword, analysis_type='lda', doc_size=doc_size, analysis_result=result_dict)

    return Response(status=status.HTTP_200_OK, data=result_dict)

# @swagger_auto_schema(
#     method='post', 
#     request_body=body_analysis_cache, responses={
#         200 : """\n\n> check_exists_only 값 true 설정 시: \n 
#         true or false
#         """ + """\n\n> check_exists 값 false 설정 시 : \n
#         {
#             "search_analysis_result": "저장된 분석 결과",  
#             "search_text_keyword": "검색문장 토크나이징 정렬 결과",
#             "analysis_type": "분석 타입",
#             "doc_size": "문서사이즈",
#             "target_db": "분석 대상 데이터 베이스",
#             "timestamp": "자료 입력 시간"
#         }  or {}""",
#         400 : 
#             APPLY_400_INVALID_TARGET_DB.as_md() +
#             APPLY_400_INVALID_ANALYSIS_TYPE.as_md() +
#             APPLY_400_SEARCH_TEXT_REQUIRED.as_md(), 
# })
# @api_view(['POST'])
def get_analysis_cache(request):
    """
    cache data 조회
    """
    search_text = request.data.get('search_text',None)
    target_db = request.data.get('target_db',None)
    is_query_string = request.data.get('is_query_string', False)
    doc_size = request.data.get('doc_size',100)
    analysis_type = request.data.get('analysis_type', None)
    check_exists_only = request.data.get('check_exists_only',False)

    # validation
    if isinstance(check_exists_only,str):
        check_exists_only = check_exists_only.upper() == 'TRUE'        
    if not search_text:
        return Response(status=APPLY_400_SEARCH_TEXT_REQUIRED.status, data={'error':APPLY_400_SEARCH_TEXT_REQUIRED.message})        
    if not analysis_type or not analysis_type in SUPPORT_ANALYSIS_TYPES:
        return Response(status=APPLY_400_INVALID_ANALYSIS_TYPE.status, data={'error':APPLY_400_INVALID_ANALYSIS_TYPE.message})        
    if isinstance(is_query_string,str):
        is_query_string = is_query_string.upper() == 'TRUE'
    
    target_db_list = target_db.split(',')
    for _target_db in target_db_list:
        if not _target_db in DOCUMENT_DB_NAMES:
            return Response(status=APPLY_400_INVALID_TARGET_DB.status, data={'error':APPLY_400_INVALID_TARGET_DB.message})
    
    # search condition
    es_query = ElasticQuery(target_db=target_db_list[0])
    
    if is_query_string:
        valid_result = es_query.validate_query_string(query_string=search_text)
        if not valid_result['valid']:
            return Response(status=APPLY_400_INVALID_QUERY_STRING.status, data={'error': valid_result['error']})
    cache_target_db, search_text_keyword = get_cache_search_format(target_db=target_db, search_text=search_text, is_query_string=is_query_string)
    target_source = ['*']
    if check_exists_only:
        target_source = ['']
    cache_data = es_query.get_analysis_cache_by_search_keys(target_db=cache_target_db, search_keys=search_text_keyword, analysis_type=analysis_type, doc_size=doc_size, source=target_source)

    if check_exists_only:
        if cache_data is None:
            return Response(status=status.HTTP_200_OK, data=False)
        return Response(status=status.HTTP_200_OK, data=True)

    if cache_data is None:
        return Response(status=status.HTTP_200_OK, data={})
    return Response(status=status.HTTP_200_OK, data=cache_data)



# @swagger_auto_schema(
#     method='post',
#     request_body=body_analysis_wordcount, 
#     responses={
#         200 : "{'차트명':차트데이터,'차트명2':차트데이터}",
#         400 : 
#             APPLY_400_DOC_IDS_REQUIRED.as_md() 
# })
# @api_view(['POST'])
def get_wordcount_chart(request):
    """
    wordcount chart 분석 
    용어의 카테고리별 + 연도별 문서 수 집계 기반 차트 데이터 생성
    -------------------
    자료구조
    * '차트타입' :  [차트데이터]
    # example
    ```
    {
        'wordcloud': [{"term":"단어","doc_freq_sum":단어등장 문서 수의 합, "norm_doc_freq_sum": 12를 최대값으로하여 doc_freq_sum 정규화한 값 },...]
    }
    ```
    """
    target_db = TARGET_DB
    doc_ids = request.data.get('doc_ids',None)
    # validation
    if not doc_ids:
        return Response(status=APPLY_400_DOC_IDS_REQUIRED.status, data={'error':APPLY_400_DOC_IDS_REQUIRED.message})
    doc_ids = doc_ids.split(',')
    sorted_doc_ids = ",".join(sorted(doc_ids))
    doc_size = len(doc_ids)
    chart_types = request.data.get('chart_types',None)
    word_limit_count = request.data.get('word_limit_count',100)
    # validation
    if not chart_types:
        return Response(status=APPLY_400_CHART_TYPE_REQUIRED.status, data={'error':APPLY_400_CHART_TYPE_REQUIRED.message})
    target_category = 'sbjt_tecl_cd'

    es_query = ElasticQuery(target_db=target_db)
    chart_type_list = chart_types.split(',')
    result_dict = {}
    # cache data 없는 chart_type 모음
    analysis_required_chart_type_list = []
    for chart_type in chart_type_list:
        if not chart_type in CHART_FUNCTION_MAPPING:
            return Response(status=APPLY_400_INVALID_CHART_TYPE.status, data={'error':APPLY_400_INVALID_CHART_TYPE.message})
        # cache data check
        cache_data = es_query.get_analysis_cache_by_search_keys(target_db=target_db, doc_ids=sorted_doc_ids, analysis_type=f'chart_{chart_type}', doc_size=doc_size)
        if cache_data:
            result_dict.update({chart_type: cache_data['analysis_result'][:int(word_limit_count)]})
            continue
        analysis_required_chart_type_list.append(chart_type)
    
    # wordcount chart analysis
    if len(analysis_required_chart_type_list) > 0:
        wc_edgelist = WordCountEdgelist()
        refined_ego_edge, term_doc_id_mapping = wc_edgelist.get_wc_refined_ego_edgelist(target_db=target_db, doc_ids=doc_ids, size=doc_size, target_category=target_category, offsets=False, positions=False)
        for chart_type in analysis_required_chart_type_list:
            if not refined_ego_edge.empty:
                if chart_type == 'wordcloud':
                    chart_result = CHART_FUNCTION_SELECTOR[chart_type](refined_ego_edge,term_doc_id_mapping)
                else:
                    chart_result = CHART_FUNCTION_SELECTOR[chart_type](refined_ego_edge)
                result_dict.update({chart_type: chart_result[:int(word_limit_count)]})
                es_query.insert_analysis_cache(target_db=target_db, search_keys=doc_ids, analysis_type=f'chart_{chart_type}', doc_size=doc_size, analysis_result=chart_result)
            else:
                result_dict.update({chart_type: ''})
    return Response(status=status.HTTP_200_OK, data=result_dict)


# @swagger_auto_schema(method='post', request_body=body_analysis_term_recommend, responses={
#         200 : """["추천복합어1","추천복합어2"]""",
#         400 : 
#             APPLY_400_INVALID_TARGET_DB.as_md() +
#             APPLY_400_SEARCH_TEXT_REQUIRED.as_md(), 
# })
# @api_view(['POST'])
def get_compound_recommend(request):
    search_text = request.data.get('search_text', None)
    is_query_string =request.data.get('is_query_string',False)
    target_db = request.data.get('target_db',None)
    doc_size = request.data.get('doc_size',100)
    term_size = request.data.get('term_size',100)
        # validation
    if not search_text:
        return Response(status=APPLY_400_SEARCH_TEXT_REQUIRED.status, data={'error':APPLY_400_SEARCH_TEXT_REQUIRED.message})
    if not target_db in DOCUMENT_DB_NAMES:
        return Response(status=APPLY_400_INVALID_TARGET_DB.status, data={'error':APPLY_400_INVALID_TARGET_DB.message})
    if isinstance(is_query_string,str):
        is_query_string = is_query_string.upper() == 'TRUE'
    result_dict = {}
    # cache result
    es_query = ElasticQuery(target_db=target_db)
    cache_target_db, search_text_keyword = get_cache_search_format(target_db=target_db, search_text=search_text, is_query_string=is_query_string)
    if is_query_string:
        valid_result = es_query.validate_query_string(query_string=search_text)
        if not valid_result['valid']:
            return Response(status=APPLY_400_INVALID_QUERY_STRING.status, data={'error': valid_result['error']})

    # edgelist cache check
    cache_edgelist = es_query.get_analysis_cache_by_search_keys(target_db=cache_target_db, search_keys=search_text_keyword, analysis_type='sna_edgelist', doc_size=doc_size)
    if cache_edgelist:
        edge_dict = cache_edgelist['analysis_result']
        ego_edge = pd.DataFrame(edge_dict['ego_edge'])
        refined_ego_edge = pd.DataFrame(edge_dict['refined_ego_edge'])
    else:
        sna_edge = SnaEdgelist()
        ego_edge, refined_ego_edge = sna_edge.get_sna_refined_ego_edgelist(target_db=target_db, search_text=search_text, is_query_string=is_query_string, size=doc_size)
        es_query.insert_analysis_cache(target_db=target_db, search_keys=search_text_keyword, analysis_type='sna_edgelist', doc_size=doc_size, analysis_result={'ego_edge':ego_edge.to_dict(),'refined_ego_edge':refined_ego_edge.to_dict()})
    compounds = EdgelistCompoundExtractor(refined_ego_edge).get_compounds(int(term_size))
    return Response(status=status.HTTP_200_OK, data={'compounds':compounds})


@api_view(['POST'])
def get_term_recommendation(request, target_db, term_type):
    """
    용어 추천
    ----
    자료 구조:
    * rec_kwd: 추천단어목록 | list(str)
    # example
    ```
    {
        "rec_kwd": ["추천단어1","추천단어2"]
    }
    ```
    """
    ...

@api_view(['POST'])
def get_player_sna(request, target_db, term_type):
    """
    플레이어 네트워크 분석
    ----
    자료 구조:
    * nodes
    id : 노드의 고유번호 | int
    t : 수행기관(section값이 None인 경우에는 주관기관) | str
    pyear : 과제 수행 연도 | list(str)
    linklen : 연결중심성 | int
    cluster : graph cluster id | int
    section : 주관기관
    doc_id : 기관 관련 과제
    
    * links
    source : source | int
    target : target | int
    v : 기관별 연결 강도(0 ~ 1) | float
    
    * player_rank : 상위 10 개 기관 | list
    doc_size : 기관 관련 문서 수 | int
    name : 기관명 | str
    weigt: 가중치 | int
    doc_id: 문서 id | list(str)
    
    # example
    ```
    {

    }
    ```
    """
    ...

@api_view(['POST'])
def get_wordcloud_chart(request, target_db):
    """
    워드클라우드 차트
    문서 내 단어의 빈도수 별로 집계
    ----
    자료 구조: 
    * wordcloud: 단어 및 해당 단어의 빈도 관련 정보 | list(dict)
        term: 단어 | str
        doc_freq_sum: 단어의 빈도수 | int
        norm_doc_freq_sum: 정규화된 단어 빈도수 | int
        doc_ids: 해당 단어가 있는 문서 id | list(str)

    # example
    ```
    {
      "wordcloud": [
        {
          "term": "단어1",
          "doc_freq_sum": 40490,
          "norm_doc_freq_sum": 12,
          "doc_ids": [
            "1465031667",
            "1345255924",
            "1711038452",
            "1711070450"
          ]
        }
    }
    ```
    """
    ...
    
@api_view(['POST'])
def get_line_chart(request, target_db):
    """
    라인 차트
    단어 / 연도별 과제 수
    
    ----
    자료 구조: 
    * word_line_chart:
        key: term(단어)
        date: 연도 list(str)
        y: 과제수 list(int)
    * paper, ipr, sbjt_line_chart
        date: 연도    
        yCount: 각 문서수
    # example
    ```
    {
      "line_chart": {
          "word_line_chart": {
            '파우치': {'date': ['2018', '2019', '2020'], 'y': [1, 1, 2]}, 
            '파우치형': {'date': ['2020'], 'y': [1]},
           },
           "paper_line_chart": {
            [{'date': '1999', 'yCount': 0}, {'date': '2000', 'yCount': 0} ...]
           },
           "ipr_line_chart": {
            [{'date': '1999', 'yCount': 0}, {'date': '2000', 'yCount': 0} ...]
           },
           "sjbt_line_chart":{
            [{'date': '1999', 'yCount': 0}, {'date': '2000', 'yCount': 0} ...]
           }
      }
    }
    ```
    """
    ...
@api_view(['POST'])
def get_bubble_chart(request, target_db):
    """
    버블차트
    각 용어가 등장하는 문서의 연도별 발생 추이
    '인공지능'이라는 단어가 전년도 대비 얼마나 증가했고, 전체 기간동안 얼마나 등장했는지 등의 정보
    상위 20개 단어들만 반환
    ----
    자료 구조: 
    * bubble_chart: 
    ** chart_data
    - id : word | str
    - year : 연도 | str
    - size : bubble의 크기(과제 수) | int
    - x : x좌표(과제 수 누적합) | int
    - y : y좌표(증가율) | float
    - t_size : 초기 선택 여부를 결정하기 위한 값, 전체 단어 등장 수 | int
    ** range_data
    - x_min : 최소 x값
    - x_max : 최대 x값
    - y_min : 최소 y값
    - y_max : 최대 y값

    # example
    ```
    {
      "bubble_chart": 
        "chart_data":{
            [{'id': 'loa',
            'year': '2004',
            'size': 2,
            'group': 'loa',
            'x': 2,
            'y': 0.0,
            't_size': 4,
            'viz': ''},
            {'id': 'loa',
            'year': '2005',
            'size': 1,
            'group': 'loa',
            'x': 3,
            'y': -50.0,
            't_size': 4,
            'viz': ''},
            {'id': 'loa',
            'year': '2006',
            'size': 1,
            'group': 'loa',
            'x': 4,
            'y': 0.0,
            't_size': 4,
            'viz': ''}]
        },
        "range_data": {
            "x_min": 0,
            "x_max": 210,
            "y_min": -1,
            "y_max": 230,
        }
    }
    ```
    """
    ...
@api_view(['POST'])
def get_bar_chart(request, target_db):
    """
    바차트
    class + 연도별 수행 과제 수
    각 연도 내 class 별로 누적막대그래프가 그려진다.
    각 연도별로 빈도가 가장 높은 5개의 class만 그려진다.
    ----
    자료 구조: 
    * bar_chart: 
    - name : section | str
    - x : 연도 | list(str)
    - y : (연도별) 과제 수 | list(int)
    - type : plotly 타입 지정 | str

    # example
    ```
    {
      "bar_chart": [
        {'name': 'E',
            'x': ['2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020'],
            'y': [11, 6, 7, 8, 5, 8, 6, 7, 32, 29, 152, 252, 404],
            'type': 'bar'
        },
        {'name': 'H', 'x': ['2011', '2013', '2014'], 'y': [1, 1, 2], 'type': 'bar'}
    }
    ```
    """
    ...

@api_view(['POST'])
def get_chart_source(request, target_db):
    """
    차트 소스 데이터
    문서 아이디를 키값으로 해당 문서의 연구비총액, 논문수, 특허수, 연도 데이터 모음
    ----
    자료 구조: 
    * chart_source: 
    - doc_id : dict key | str
    - publication_datex : 연도 | str
    - rndco_tot_amt : 연구비 총액 | int
    - ipr :  특허 수 | int
    - paper : 논문 수 | int

    # example
    ```
  {
      "chart_source": {
            "1345235770": {
                "date": "2015",
                "rndco_tot_amt": 13000000,
                "ipr": 0,
                "paper": 1
            }
        }
    }
    ```
    """
    ...
