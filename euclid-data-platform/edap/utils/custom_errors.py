from rest_framework import status
from config.constants import DOCUMENT_DB_NAMES, TERM_TYPES, SUPPORT_ANALYSIS_TYPES, CHART_FUNCTION_MAPPING, MAX_CLAUSE_COUNT


class ErrorCollection(object):

    def __init__(self, code, status, message):
        self.code = code
        self.status = status
        self.message = message

    def set_msg(self, message):
        self.message = message

    def as_md(self):
        return '\n\n> **%s**\n\n```\n{\n\n\t"code": "%s"\n\n\t"message": "%s"\n\n}\n\n```' % \
               (self.message, self.code, self.message)

class FailedQueryParsing(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return f"[FailedQueryParsing]: failed to parse querystring : {self.msg}"

class TooManyClause(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return f"[TooManyClause]: too many clause it must under {MAX_CLAUSE_COUNT} but it has {self.msg}"


APPLY_400_FULL_QUERY_REQUIRED = ErrorCollection(
    code='APPLY_400_FULL_QUERY_REQUIRED',
    status=status.HTTP_400_BAD_REQUEST,
    message='full query required'
)
APPLY_400_DOC_IDS_REQUIRED = ErrorCollection(
    code='APPLY_400_DOC_IDS_REQUIRED',
    status=status.HTTP_400_BAD_REQUEST,
    message='doc_ids required'
)
APPLY_400_SEARCH_FORM_REQUIRED = ErrorCollection(
    code='APPLY_400_SEARCH_FORM_REQUIRED',
    status=status.HTTP_400_BAD_REQUEST,
    message='search_form required'
)

APPLY_400_INVALID_QUERY_STRING = ErrorCollection(
    code='APPLY_400_INVALID_QUERY_STRING',
    status=status.HTTP_400_BAD_REQUEST,
    message='invalid query string'
)

APPLY_400_INVALID_DOC_SIZE = ErrorCollection(
    code='APPLY_400_INVALID_DOC_SIZE',
    status=status.HTTP_400_BAD_REQUEST,
    message='invalid doc size, doc size is between 0~1000'
)

APPLY_400_INVALID_TARGET_DB = ErrorCollection(
    code='APPLY_400_INVALID_TARGET_DB',
    status=status.HTTP_400_BAD_REQUEST,
    message=f'database is not in {DOCUMENT_DB_NAMES}'
)

APPLY_400_INVALID_ANALYSIS_TYPE = ErrorCollection(
    code='APPLY_400_INVALID_ANALYSIS_TYPE',
    status=status.HTTP_400_BAD_REQUEST,
    message=f'analysis_type is not in {SUPPORT_ANALYSIS_TYPES}'
)
APPLY_400_INVALID_CHART_TYPE = ErrorCollection(
    code='APPLY_400_INVALID_CHART_TYPE',
    status=status.HTTP_400_BAD_REQUEST,
    message=f'chart_type is not in {list(CHART_FUNCTION_MAPPING.keys())}'
)

APPLY_400_INVALID_TERM_TYPE = ErrorCollection(
    code='APPLY_400_INVALID_TERM_TYPE',
    status=status.HTTP_400_BAD_REQUEST,
    message=f'term_type is not in {TERM_TYPES.values()}'
)

APPLY_400_INVALID_TARGET_TERMS = ErrorCollection(
    code='APPLY_400_INVALID_TARGET_TERMS',
    status=status.HTTP_400_BAD_REQUEST,
    message=f'inappropriate target_terms format, should be array-like object'
)

APPLY_400_INVALID_PK = ErrorCollection(
    code='APPLY_400_INVALID_PK',
    status=status.HTTP_400_BAD_REQUEST,
    message=f'invalid pk'
)

APPLY_400_TARGET_IDS_REQUIRED = ErrorCollection(
    code='APPLY_400_TARGET_IDS_REQUIRED',
    status=status.HTTP_400_BAD_REQUEST,
    message=f'target_ids required'
)

APPLY_400_TARGET_DBS_REQUIRED = ErrorCollection(
    code='APPLY_400_TARGET_DBS_REQUIRED',
    status=status.HTTP_400_BAD_REQUEST,
    message=f'target_dbs required'
)

APPLY_400_TARGET_TERMS_REQUIRED = ErrorCollection(
    code='APPLY_400_TARGET_TERMS_REQUIRED',
    status=status.HTTP_400_BAD_REQUEST,
    message=f'target_terms required'
)
APPLY_400_CHART_TYPE_REQUIRED = ErrorCollection(
    code='APPLY_400_CHART_TYPE_REQUIRED',
    status=status.HTTP_400_BAD_REQUEST,
    message='chart type required'
)

APPLY_400_SEARCH_TEXT_REQUIRED = ErrorCollection(
    code='APPLY_400_SEARCH_TEXT_REQUIRED',
    status=status.HTTP_400_BAD_REQUEST,
    message=f'search_text required'
)
