from django.db.models.fields import Field
from django.db.models import Lookup
from django.conf import settings
from documents.models import Document
from terms.models import Terminology

# 공통 사용 상수들

# Limenet_analysis target db
TARGET_DB = 'kistep_sbjt'

# 사용하는 데이터베이스 정보 (conf/settings/ 아래 DATABASES 항목들)
DATABASES = settings.DATABASES

# 문서 검색 결과 제한
DOC_SIZE_LIMIT = 2000

# 클라우드 차트 단어 개수 제한
CLOUD_CHART_WORD_CNT_LIMIT = 200

# 검색 조건 토큰 수 제한
MAX_CLAUSE_COUNT = 2048

# 문서가 저장되는 테이블 이름
DOCUMENT_TABLE_NAME = Document.objects.model._meta.db_table

# 문서 db 이름 리스트
DOCUMENT_DB_NAMES = []

# 엘라스틱서치 클러스터 정보
ELASTICSEARCH_CLUSTER_INFO = {}

if len(DOCUMENT_DB_NAMES) == 0 and len(ELASTICSEARCH_CLUSTER_INFO) == 0:
    for db_name in DATABASES:
        if db_name == 'kistep_sbjt' or db_name == 'test_db':
            DOCUMENT_DB_NAMES.append(db_name)
            ELASTICSEARCH_CLUSTER_INFO.update(
                {db_name: DATABASES[db_name]['ELASTICSEARCH_INFO']['CLUSTER_SERVER_INFO']})

# Term 관련 용어 정의(용어 관련 용어명이 변경 될 경우 value값 변경, 변경 시 검증 작업 안되어있으므로 가급적 변경하지 말고 사용할 것)
TERM_TYPES = {
    'compound': 'compound',
    'edgelist_term': 'edgelist_term',
    'representative': 'representative',
    'synonym': 'synonym',
    'stopword': 'stopword',
    'terminology': 'terminology',
}

# 명사 저장 테이블 이름
TERM_TABLE_NAME = Terminology.objects.model._meta.db_table

# secret folder path
SECRET_FOLDER_PATH = str(settings.BASE_DIR)+'/config/settings/secret'

# 분석 기능
SUPPORT_ANALYSIS_TYPES = [
    'sna',
    'lda',
    'player_sna',
    'rec_kwd',
]

# 차트 생성 로직 모듈 경로
CHART_MODULE_PATH = 'analysis.wordcount.d3_chartdata'

# 차트 타입 별 생성 함수 이름
CHART_FUNCTION_MAPPING = {
    'wordcloud': 'make_cloud_chart',
    'bubble_chart': 'bubble_chart',
    'bar_chart': 'bar_chart',
    'line_chart': 'line_chart_ver3',
    'chart_source': 'chart_source_map',
}
SUPPORT_ANALYSIS_TYPES += list(CHART_FUNCTION_MAPPING.keys())
# source fields for elasticsearch
ES_DOC_SOURCE = [
    'doc_id',
    'pjt_id',
    'stan_yr',
    'yr_cnt',
    'hm_nm',
    'rsch_area_cls_cd',
    'rsch_area_cls_nm',
    'kor_kywd',
    'eng_kywd',
    'rsch_goal_abstract',
    'rsch_abstract',
    'exp_efct_abstract',
    'pjt_prfrm_org_cd',
    'pjt_prfrm_org_nm',
    'kor_pjt_nm',
    'eng_pjt_nm',
    'pjt_mgnt_org_cd',
    'spclty_org_nm',
    'prctuse_nm',
    'rnd_phase',
    'rsch_exec_suj',
    'dtl_pjt_clas',
    'tech_lifecyc_nm',
    'regn_nm',
    'pjt_no',
    'tot_rsch_start_dt',
    'tot_rsch_end_dt',
    'tsyr_rsch_start_dt',
    'tsyr_rsch_end_dt',
    'rndco_tot_amt',
    "ipr",
    "paper",
 ]

# Edgelist
BASE_EDGELIST_COLUMN_MAPPING = {
    'doc_id': 'doc_id',
    'subclass': 'doc_subclass',
    'title': 'kor_pjt_nm',
    'publication_date': 'stan_yr',
    'pjt_mgnt_org_cd': 'pjt_mgnt_org_cd',
    'spclty_org_nm': 'spclty_org_nm',
    'pjt_prfrm_org_cd': 'pjt_prfrm_org_cd',
    'pjt_prfrm_org_nm': 'pjt_prfrm_org_nm',
    'rndco_tot_amt': 'rndco_tot_amt',
    'ipr': 'ipr',
    'paper': 'paper',
    'hm_nm' : 'hm_nm',
}

SNA_EDGELIST_COLUMN_MAPPING = {
    'doc_id': 'doc_id',
    'index': 'DELETE',
    'term': 'word',
    'field': 'col_type',
    'doc_freq': 'DELETE',
    'ttf': 'DELETE',
    'term_freq': 'dtfreq',
    'position': 'position',
    'token_info': 'DELETE',
    'section': 'section',
    'subclass': 'd_class',
    'score': 'DELETE',
    'title': 'title',
    'publication_date': 'pyear',
    'pjt_mgnt_org_cd': 'DELETE',
    'spclty_org_nm': 'DELETE',
    'pjt_prfrm_org_cd': 'DELETE',
    'pjt_prfrm_org_nm': 'DELETE',
}

WORDCOUNT_EDGELIST_COLUMN_MAPPING = {
    'doc_id': 'doc_id',
    'index': 'DELETE',
    'term': 'term',
    'field': 'DELETE',
    'doc_freq': 'DELETE',
    'ttf': 'DELETE',
    'term_freq': 'term_freq',
    'position': 'DELETE',
    'token_info': 'DELETE',
    'section': 'DELETE',
    'subclass': 'subclass',
    'score': 'DELETE',
    'title': 'DELETE',
    'publication_date': 'publication_date',
    'pjt_mgnt_org_cd': 'DELETE',
    'spclty_org_nm': 'DELETE',
    'pjt_prfrm_org_cd': 'DELETE',
    'pjt_prfrm_org_nm': 'DELETE',
    'rndco_tot_amt': 'rndco_tot_amt',
    'ipr': 'ipr',
    'paper': 'paper',
}

SUBJECT_PLAYER_EDGELIST_COLUMN_MAPPING = {
    'doc_id': 'doc_id',
    'index': 'DELETE',
    'term': 'term',
    'field': 'DELETE',
    'doc_freq': 'DELETE',
    'ttf': 'DELETE',
    'term_freq': 'DELETE',
    'position': 'DELETE',
    'token_info': 'DELETE',
    'section': 'doc_section',
    'subclass': 'subclass',
    'score': 'DELETE',
    'title': 'title',
    'publication_date': 'pyear',
    'pjt_mgnt_org_cd': 'mgnt_org_cd',
    'spclty_org_nm': 'mgnt_org_nm',
    'pjt_prfrm_org_cd': 'prfrm_org_cd',
    'pjt_prfrm_org_nm': 'prfrm_org_nm',
    'hm_nm' : 'hm_nm',
}

# Like 검색 필터


@Field.register_lookup
class Like(Lookup):
    lookup_name = 'like'

    def as_sql(self, compiler, connection):
        lhs, lhs_params = self.process_lhs(compiler, connection)
        rhs, rhs_params = self.process_rhs(compiler, connection)
        params = lhs_params + rhs_params
        return '%s LIKE %s' % (lhs, rhs), params
