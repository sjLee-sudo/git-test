import json
import timeit
from typing import OrderedDict
from django.core.paginator import InvalidPage, Paginator
from django.db.models import Q
from django.core import serializers
from django.core.cache import cache
from collections.abc import Iterable

from rest_framework.exceptions import NotFound
from django.utils.functional import cached_property
from config.constants import DOCUMENT_DB_NAMES, TERM_TYPES
from db_manager.managers import ElasticManager, ElasticQuery
from terms.userdict import create_copy_userdict
from terms.term_utils import DOCU_TERM_MAPPER, create_stopword_from_word_list, create_terminology_from_word_list, create_edgelist_term_from_word_list, create_compound_from_word_list, validate_compound,create_synonym_and_representative_from_word_list, validate_sub_term_exists, validate_unhashable_type, validate_ids
from utils.custom_errors import APPLY_400_INVALID_TARGET_TERMS, APPLY_400_INVALID_TARGET_DB, APPLY_400_INVALID_TERM_TYPE, APPLY_400_INVALID_PK, APPLY_400_TARGET_IDS_REQUIRED, APPLY_400_TARGET_TERMS_REQUIRED
from utils.common_utils import get_filter

from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework.pagination import PageNumberPagination

from drf_yasg import openapi 
from drf_yasg.utils import swagger_auto_schema

param_target_terms = openapi.Parameter(
    'target_terms',
    openapi.IN_QUERY,
    description='해당하는 단어만 조회. comma(,)로 구분해서 요청.',
    type=openapi.TYPE_STRING
)
param_filter_type = openapi.Parameter(
    'filter_type',
    openapi.IN_QUERY,
    example='in, like',
    description='단어 목록 필터링 조건, 기본값 \"in\". 지원하지 않는 filter type은 \"in\" 조건으로 변경됨 \n \"like\" 필터타입은 \% 연산자 사용 가능',
    type=openapi.TYPE_STRING
)
param_page_size = openapi.Parameter(
    'page_size',
    openapi.IN_QUERY,
    description='한 페이지에 보여줄 데이터 수',
    type=openapi.TYPE_STRING
)
param_page = openapi.Parameter(
    'page',
    openapi.IN_QUERY,
    description='페이지 번호',
    type=openapi.TYPE_STRING
)
body_target_terms = openapi.Schema(
    type=openapi.TYPE_OBJECT,
    required=['target_terms'],
    example={
        'target_terms(term_type 이 terminoloy, stopword, edgelist_term 일 경우)':['용어1','용어2'],
        'target_terms(term_type 이 synonym, representative, compound) 일 경우)':[['용어1','용어1의 동의어'],['대표어1', '대표어1의 변환대상용어'], ['복합어1(공백없음)', '복합어의 구성 용어1','복합어의 구성 용어2']],
        },
    properties={
        'targe_terms': openapi.Schema(
                type=openapi.TYPE_ARRAY,
                description='추가할 용어 목록 List',
                items={'type':openapi.TYPE_STRING}
            ),
    }
)
body_target_ids = openapi.Schema(
    type=openapi.TYPE_OBJECT,
    required=['target_ids'],
    example={
        'target_ids':[1,2,3],
        },
    properties={
        'targe_ids': openapi.Schema(
                type=openapi.TYPE_ARRAY,
                description='삭제할 용어의 pk 목록 List',
                items={'type':openapi.TYPE_INTEGER}
            ),
    }
)
path_target_db = openapi.Parameter(
    'target_db',
    openapi.IN_PATH,
    example = ", ".join(DOCUMENT_DB_NAMES),
    description= f'데이터베이스명',
    type=openapi.TYPE_STRING
)
path_term_type = openapi.Parameter(
    'term_type',
    openapi.IN_PATH,
    example=", ".join(TERM_TYPES.values()),
    description= f'용어타입',
    type=openapi.TYPE_STRING
)

class FasterDjangoPaginator(Paginator):
    @cached_property
    def count(self):
        return self.object_list.values('id').count()

class CustomPaginator(PageNumberPagination):

    def paginate_queryset(self, queryset, request, view=None):
        page_size = self.get_page_size(request)
        if not page_size:
            return None

        paginator = FasterDjangoPaginator(queryset, page_size)
        page_number = request.query_params.get(self.page_query_param, 1)

        try:
            self.page = paginator.page(page_number)
        except InvalidPage as e:
            msg = self.invalid_page_message.format(
                page_number=page_number, message=str(e)
            )
            raise NotFound(msg)
        if paginator.num_pages > 1 and self.template is not None:
            # The browsable API should display pagination controls.
            self.display_page_controls = True
        self.request = request        
        return list(self.page)

    def get_paginated_response(self, data):
        return Response(OrderedDict([
                ('count', self.page.paginator.count),
                ('next', self.get_next_link()),
                ('previous', self.get_previous_link()),
                ('total_pages', self.page.paginator.num_pages),
                ('results', data)
            ])
        )
            

class TermListView(APIView):
    """
    용어 CRUD API
    """
    paginator = CustomPaginator()

    @swagger_auto_schema(manual_parameters=[path_target_db, path_term_type, param_page_size, param_page, param_target_terms, param_filter_type],responses={
        200 : '{"count":데이터수, "next": 다음페이지 url, "previous":이전페이지 url, "total_pages": 전체페이지 수, "results": 데이터 }',
        400 : 
            APPLY_400_INVALID_TARGET_DB.as_md() +
            APPLY_400_INVALID_TERM_TYPE.as_md(), 
    })
    def get(self, request, target_db, term_type):
        """
        용어 리스트 API
        """
        if not target_db in DOCUMENT_DB_NAMES:
            return Response(status=APPLY_400_INVALID_TARGET_DB.status, data={'error':APPLY_400_INVALID_TARGET_DB.message})
        if not term_type in TERM_TYPES.values():
            return Response(status=APPLY_400_INVALID_TERM_TYPE.status, data={'error':APPLY_400_INVALID_TERM_TYPE.message})

        filter_value = request.query_params.get('target_terms')
        filter_type = request.query_params.get('filter_type')
        queryset = DOCU_TERM_MAPPER[term_type]['model'].objects.using(target_db)
        if filter_value is not None:
            if filter_type is None or not filter_type.lower() in ['in','like','rlike','contains']:
                filter_type = 'in'
            filter_type = filter_type.lower()
            if filter_type == 'in':
                _filter_value = filter_value.split(',')
                _striped_filter_value = [ val.replace(' ','') for val in _filter_value]
                filter_value = list(set(_filter_value + _striped_filter_value))
            if term_type == TERM_TYPES['synonym'] or term_type == TERM_TYPES['representative']:
                queryset = queryset.filter(get_filter('main_term',filter_type,filter_value)|get_filter('sub_term',filter_type,filter_value)).order_by('main_term','sub_term')
                # queryset = queryset.filter(Q(main_term__term__in=target_term_list)|Q(sub_term__term__in=target_term_list)).select_related('main_term','sub_term')
            elif term_type == TERM_TYPES['stopword']:
                queryset = queryset.filter(get_filter('stopword',filter_type,filter_value)).order_by('stopword')
                # queryset = queryset.filter(stopword__in=target_term_list)
            elif term_type == TERM_TYPES['compound']:
                queryset = queryset.filter(get_filter('compound',filter_type,filter_value)).order_by('compound','component')
                # queryset = queryset.filter(compound__in=target_term_list)
            else:
                queryset = queryset.filter(get_filter('term',filter_type,filter_value))
                # queryset = queryset.filter(term__in=target_term_list)
        else:
            queryset = queryset.all()
        self.paginator.page_size_query_param = 'page_size'
        result_page = self.paginator.paginate_queryset(queryset=queryset, request=request,view=self)
        if result_page is not None:
            serializer = self.paginator.get_paginated_response(DOCU_TERM_MAPPER[term_type]['serializer'](result_page, many=True).data)
        else:
            serializer = DOCU_TERM_MAPPER[term_type]['serializer'](queryset, many=True)
        return Response(status=status.HTTP_200_OK, data=serializer.data)


    @swagger_auto_schema(manual_parameters=[path_target_db,path_term_type],
        request_body=body_target_terms, responses={
        200 : '{"success":[{입력된 용어타입 및 상세 정보}],fail:[실패한 데이터 및 원인]}',
        400 : 
            APPLY_400_INVALID_TARGET_DB.as_md() +
            APPLY_400_INVALID_TERM_TYPE.as_md() +
            APPLY_400_TARGET_TERMS_REQUIRED.as_md() +
            APPLY_400_INVALID_TARGET_TERMS.as_md(), 
    })
    def post(self, request, target_db, term_type):
        """
        용어 다중 입력 API
        """
        if not target_db in DOCUMENT_DB_NAMES:
            return Response(status=APPLY_400_INVALID_TARGET_DB.status, data={'error':APPLY_400_INVALID_TARGET_DB.message})
        if not term_type in TERM_TYPES:
            return Response(status=APPLY_400_INVALID_TERM_TYPE.status, data= {'error': APPLY_400_INVALID_TERM_TYPE})
        if not 'target_terms' in request.data or not request.data['target_terms']:
            return Response(status=APPLY_400_TARGET_TERMS_REQUIRED.status, data={'error':APPLY_400_TARGET_TERMS_REQUIRED.message})
        
        target_term_list = request.data['target_terms']
        
        if not isinstance(target_term_list,Iterable) or isinstance(target_term_list,str):
            return Response(status=APPLY_400_INVALID_TARGET_TERMS.status, data={'error':APPLY_400_INVALID_TARGET_TERMS.message})
        
        result_dict = {'success':'','fail':''}
        
        res = []
        error_list = []

        if term_type == TERM_TYPES['compound']:
            target_term_list, error_list = validate_compound(target_term_list)
            if len(target_term_list) >0:
                res = create_compound_from_word_list(target_db=target_db, compound_list=target_term_list)
        elif term_type == TERM_TYPES['synonym']:
            target_term_list, error_list = validate_sub_term_exists(target_term_list)
            if len(target_term_list) > 0:
                res, error_list = create_synonym_and_representative_from_word_list(target_db=target_db, main_sub_term_list=target_term_list, term_type='synonym')
        elif term_type == TERM_TYPES['representative']:
            target_term_list, error_list = validate_sub_term_exists(target_term_list)
            if len(target_term_list) > 0:
                res, error_list = create_synonym_and_representative_from_word_list(target_db=target_db, main_sub_term_list=target_term_list, term_type='representative')
        elif term_type == TERM_TYPES['stopword']:
            target_term_list, error_list= validate_unhashable_type(target_term_list)
            if len(target_term_list) > 0:
                res = create_stopword_from_word_list(target_db=target_db, word_list=target_term_list)
        elif term_type == TERM_TYPES['terminology']:
            target_term_list, error_list= validate_unhashable_type(target_term_list)
            if len(target_term_list) > 0:
                res = create_terminology_from_word_list(target_db=target_db, word_list=target_term_list)
        elif term_type == TERM_TYPES['edgelist_term']:
            target_term_list, error_list= validate_unhashable_type(target_term_list)
            if len(target_term_list) > 0:
                res = create_edgelist_term_from_word_list(target_db=target_db, word_list=target_term_list)
        error_res = {'inappropriate data structure': error_list}
        json_res = serializers.serialize('json',res)
        res = json.loads(json_res)
        result_dict.update({'success':res, 'fail': error_res})
        return Response(status=status.HTTP_200_OK, data=result_dict)
    
    @swagger_auto_schema(manual_parameters=[path_target_db,path_term_type], request_body=body_target_ids, responses={
        200 : '{"success":[삭제된 데이터개수, {용어타입}],fail:[실패한 데이터 수 및 원인]}',
        400 : 
            APPLY_400_INVALID_TARGET_DB.as_md() +
            APPLY_400_INVALID_TERM_TYPE.as_md() +
            APPLY_400_TARGET_IDS_REQUIRED.as_md(), 
    })
    def delete(self, request, target_db, term_type):
        """
        용어 다중 삭제 API
        """
        if not target_db in DOCUMENT_DB_NAMES:
            return Response(status=APPLY_400_INVALID_TARGET_DB.status, data={'error':APPLY_400_INVALID_TARGET_DB.message})
        if not 'target_ids' in request.data or not request.data['target_ids']:
            return Response(status=APPLY_400_TARGET_IDS_REQUIRED.status, data={'error':APPLY_400_TARGET_IDS_REQUIRED.message})
        target_id_list = request.data['target_ids']
        target_id_list, error_list = validate_ids(target_id_list)
        delete_res = {}
        delete_target_terms = []
        if len(target_id_list) > 0:
            delete_target_objs = DOCU_TERM_MAPPER[term_type]['model'].objects.using(target_db).filter(id__in=target_id_list)
            if term_type == 'edgelist_term':
                delete_target_terms = list(delete_target_objs.values_list('term',flat=True))
                # cahce data delete
                cache.delete(f'{target_db}_edgelist_term')
                es_query = ElasticQuery(target_db)
                es_query.delete_data_by_field_terms(index_name='analysis_cache', field_name='search_text_keyword', terms=delete_target_terms)
            delete_res = delete_target_objs.delete()
        result_dict = {'success': delete_res,'fail':error_list}
        return Response(status=status.HTTP_200_OK, data=result_dict)

class TermView(APIView):
    @swagger_auto_schema(manual_parameters=[path_target_db,path_term_type], responses={
        200 : '{용어id 및 컬럼명, 용어}',
        400 : 
            APPLY_400_INVALID_TARGET_DB.as_md() +
            APPLY_400_INVALID_PK.as_md() ,
    })
    def get(self, request, target_db, term_type, pk,):
        """
        pk로 용어 조회 API
        """
        if not target_db or not term_type:
            return Response(status=APPLY_400_INVALID_TARGET_DB.status, data={'error':APPLY_400_INVALID_TARGET_DB.message})
        try:
            queryset = DOCU_TERM_MAPPER[term_type]['model'].objects.using(target_db).get(pk=pk)
        except DOCU_TERM_MAPPER[term_type]['model'].DoesNotExist:
            return Response(status=APPLY_400_INVALID_PK.status, data={'error':APPLY_400_INVALID_PK.message})
        serializer = DOCU_TERM_MAPPER[term_type]['serializer'](queryset)
        return Response(status=status.HTTP_200_OK, data=serializer.data)

    @swagger_auto_schema(manual_parameters=[path_target_db,path_term_type], responses={
        200 : '''{"success":[삭제된 용어개수,{"용어모델":개수}],"fail":""}\n```\n{\n\n\t"success": [1,{"terms.Synonym":1}],"fail":""\n\n}\n```''',
        400 : 
            APPLY_400_INVALID_TARGET_DB.as_md() +
            APPLY_400_INVALID_PK.as_md() ,
    })
    def delete(self, request, target_db, term_type, pk):
        """
        pk로 용어 삭제 API
        """
        if not target_db or not term_type:
            return Response(status=APPLY_400_INVALID_TARGET_DB.status, data={'error':APPLY_400_INVALID_TARGET_DB.message})
        try:
            queryset = DOCU_TERM_MAPPER[term_type]['model'].objects.using(target_db).get(pk=pk)
        except DOCU_TERM_MAPPER[term_type]['model'].DoesNotExist:
            return Response(status=APPLY_400_INVALID_PK.status, data={'error':APPLY_400_INVALID_PK.message})
        # cache data delete
        if term_type == 'edgelist_term':
            delete_term = [queryset.term]
            cache.delete(f'{target_db}_edgelist_term')
            es_query = ElasticQuery(target_db)
            es_query.delete_data_by_field_terms(index_name='analysis_cache', field_name='search_text_keyword', terms=delete_term)
        res = queryset.delete()
        result_dict = {'success': res,'fail': ''}
        return Response(status=status.HTTP_200_OK, data=result_dict)


@api_view(['GET'])
def make_userdict(request, target_db, term_type):
    """
    #사용주의!
    ----
    개발용 사용자사전 반영 API
    해당 기능 이용 시 엘라스틱서치 인덱스 삭제 등 위험성이 크고, 다중 요청에 안전하지 않으므로
    운영환경에서는 다른 방식으로 기능 제공 예정
    """
    start_time = timeit.default_timer()
    if term_type != 'all':
        if not term_type in TERM_TYPES.values():
            return Response(status=APPLY_400_INVALID_TERM_TYPE.status, data={'error':APPLY_400_INVALID_TERM_TYPE.message})

    if target_db != 'all':
        if not target_db in DOCUMENT_DB_NAMES:
            return Response(status=APPLY_400_INVALID_TARGET_DB.status, data={'error':APPLY_400_INVALID_TARGET_DB.message})

    target_db_list = [target_db]
    target_term_type_list = [term_type]
    if target_db == 'all':
        target_db_list = DOCUMENT_DB_NAMES
    if term_type == 'all':
        target_term_type_list = list(TERM_TYPES.values())
    result = []
    create_copy_userdict(target_db_list, target_term_type_list)
    for _target_db in target_db_list:
        es_manager = ElasticManager(target_db=_target_db)
        for _term_type in target_term_type_list:
            # dictionary file create
            # copy dictionary file to elasticsearch
            create_result, copy_result= create_copy_userdict(target_db=_target_db, term_type=_term_type)
            result.append(create_result)
            result.append(copy_result)
        reindex_res = es_manager.reindexing()
        result.append(reindex_res)
    return Response(status=status.HTTP_200_OK, data={'message':'finished','detail':result,'took':f'{timeit.default_timer()-start_time}'})

@api_view(['GET'])
def get_nstc(request):
    """
    
    """
    