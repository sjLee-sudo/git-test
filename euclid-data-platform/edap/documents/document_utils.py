import pandas as pd
import re

if __name__ != '__main__':
    from db_manager.managers import ElasticQuery
    from utils.custom_errors import TooManyClause
    from config.constants import ES_DOC_SOURCE, MAX_CLAUSE_COUNT, TARGET_DB
    from terms.userdict_utils import word_only_pattern

reserved_word_pattern = re.compile(r'[\+\|\-\"\*\(\)\^\!\/\[\]]|~\d')

es_query = ElasticQuery(target_db=TARGET_DB)

def create_topic_model_search_query(search_form, fields=None, source=['*'], use_simple_query=False):
    if not fields:
        fields = ["analysis_target_text"]
    if use_simple_query:
        query = {
            "_source": source,
            "sort": [
                {
                    "_score": {
                        "order": "desc"
                    }
                },
                {
                    "stan_yr": {
                        "order": "desc"
                    }
                }
                
            ],
            "query":{
                "bool":{
                    "must":[
                        {
                            "simple_query_string":{
                                "fields": fields,
                                "query": search_form["query_string"]
                            }
                        }
                    ]
                }
            }
        }
    else:
        query = {
            "_source": source,
            "sort": [
                {
                    "_score": {
                        "order": "desc"
                    }
                },
                {
                    "stan_yr": {
                        "order": "desc"
                    }
                }
                
            ],
            "query":{
                "bool":{
                    "must":[
                        {
                            "query_string":{
                                "fields": fields,
                                "query": search_form["query_string"]
                            }
                        }
                    ]
                }
            }
        }

    if "filter" in search_form and len(search_form["filter"]) > 0:
        filter_form = {"filter": []}
        for _filter in search_form["filter"]:
            for _field, value in _filter.items():
                filter_form["filter"].append({"terms": {_field:value}})
        query["query"]["bool"].update(filter_form)    
    return query

def create_keyword_search_query(search_form, fields=None, source=['*'], use_simple_query=False):
    """
    예약어 처리 필요
    + signifies AND operation
    | signifies OR operation
    - negates a single token
    " wraps a number of tokens to signify a phrase for searching
    * at the end of a term signifies a prefix query
    ( and ) signify precedence
    ~N after a word signifies edit distance (fuzziness)
    ~N after a phrase signifies slop amount
    """
    if not fields:
        fields = ["kor_kywd^1.5","eng_kywd^1.5","kor_pjt_nm^1.2","rsch_goal_abstract","rsch_abstract","exp_efct_abstract","analysis_target_text"]
    total_clause = 0
    combined_query = ""
    default_boost = 1.2
    or_text = []
    exclude_text = []
    not_query = ''
    or_query = ''
    if 'search_text' in search_form:
        if not isinstance(search_form['search_text'],list):
            search_form['search_text'] = str(search_form['search_text']).split(',')
        for _search_text in search_form['search_text']:
            if len(_search_text) == 0:
                continue
            _search_text = word_only_pattern.sub('',_search_text)
            or_text.append(_search_text)
            if _search_text.find(" ") > 0:
                phrase_query = '\"'+_search_text+'\"' +f'^{default_boost}'
                or_text.append(phrase_query)
    if 'ordered_text' in search_form:
        if not isinstance(search_form['ordered_text'],list):
            search_form['ordered_text'] = str(search_form['ordered_text']).split(',')
        ordered_text_size = len(search_form['ordered_text'])
        for idx, _ordered_text in enumerate(search_form['ordered_text']):
            if len(_ordered_text) == 0:
                continue
            _ordered_text = word_only_pattern.sub('',_ordered_text)        
            boost_value = round(((ordered_text_size-idx)/ordered_text_size) + 1,2)
            if _ordered_text.find(" ") > 0:
                boost_value = round(default_boost + boost_value,2)
                phrase_query = '\"'+_ordered_text+'\"'+f'^{boost_value}'
                or_text.append(phrase_query)
            else:
                or_text.append(_ordered_text+f"^{boost_value}")
    if 'exclude_text' in search_form:
        if not isinstance(search_form['exclude_text'],list):
            search_form['exclude_text'] = str(search_form['exclude_text']).split(',')
        for _exclude_text in search_form['exclude_text']:
            if len(_exclude_text) == 0:
                continue
            _exclude_text = word_only_pattern.sub('',_exclude_text)
            exclude_text.append(_exclude_text)
    
    total_clause += len(or_text)
    total_clause += len(exclude_text)
    # 조건 토큰 제한 
    if total_clause >= MAX_CLAUSE_COUNT :
        raise TooManyClause(total_clause)

    or_query = "|".join(or_text)
    not_query = "|".join(exclude_text)
    if len(not_query)>0:
        combined_query = f"({or_query}) + -({not_query})"
    else:
        combined_query = or_query
    if use_simple_query:
        query = {
            "_source": source,
            "sort": [
                {
                    "_score": {
                        "order": "desc"
                    }
                },
                    {
                    "stan_yr": {
                        "order": "desc"
                    }
                }
                
            ],
            "query":{
                "bool":{
                    "must":[
                        {
                            "simple_query_string":{
                                "fields": fields,
                                "query": combined_query
                            }
                        }
                    ]
                }
            }
        }
    else:
        query = {
            "_source": source,
            "sort": [
                {
                    "_score": {
                        "order": "desc"
                    }
                },
                {
                    "stan_yr": {
                        "order": "desc"
                    }
                }
                
            ],
            "query":{
                "bool":{
                    "must":[
                        {
                            "query_string":{
                                "fields": fields,
                                "query": combined_query
                            }
                        }
                    ]
                }
            }
        }
    if "filter" in search_form and len(search_form["filter"]) > 0:
        filter_form = {"filter": []}
        for _filter in search_form["filter"]:
            for _field, value in _filter.items():
                filter_form["filter"].append({"terms": {_field:value}})
        query["query"]["bool"].update(filter_form)
    return query


def create_subject_search_query(target_subject, source=['*'], use_simple_query=False):
    query = {
        "_source": source,
        "sort": [
                {
                    "_score": {
                        "order": "desc"
                    }
                },
                {
                    "stan_yr": {
                        "order": "desc"
                    }
                }
                
        ],
        "query": {
            "bool":{
                "should": [

                ]
            }
        }
    }
    for field, value in target_subject.items():
        # try:
        if not field in ES_DOC_SOURCE:
            continue
        if value is None or value =="" or not value:
            continue
        if field in ["rndco_tot_amt"]:
            try:
                query["query"]["bool"]["should"].append(
                    {
                        "range":{
                            field: {
                                "gte": int(value) * 1.2,
                                "lte": int(value) * 0.8,
                            }
                        }
                    }
                )
            except ValueError:
                continue
            continue
        if field =='kor_kywd' or field =='eng_kywd':
            field = field + "^1.5"
            if isinstance(value, list):
                value = ",".join(value)
        if not isinstance(value,str):
            value = str(value)
        value = word_only_pattern.sub('',value)

        if value == "":
            continue
        # clause token limit check
        if len(value) > MAX_CLAUSE_COUNT:
            value = es_query.trim_text_under_max_clauses_size(search_text=value,field=field)

        if use_simple_query:
            query["query"]["bool"]["should"].append(
                {
                    "simple_query_string": {
                        "fields": [field],
                        "query": value
                    }   
                }
            )
        else:
            query["query"]["bool"]["should"].append(
                {
                    "query_string": {
                        "fields": [field],
                        "query": value
                    }   
                }
            )
            
        # except TypeError:
        #     continue
    return query



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
    from db_manager.managers import ElasticQuery
    from utils.custom_errors import TooManyClause
    from config.constants import ES_DOC_SOURCE, MAX_CLAUSE_COUNT, TARGET_DB
    from terms.userdict_utils import word_only_pattern