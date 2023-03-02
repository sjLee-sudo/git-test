import pandas as pd
from django.db.models import Q
from collections.abc import Iterable
from django.core.cache import cache
import re
import os, datetime

def initialize_term_model():
    term_model_mapping = {
        'terminology': {
            'model': Terminology,
            'serializer': TerminologySerializer,
        },
        'compound': {
            'model': Compound,
            'serializer': CompoundSerializer
        },
        'stopword': {
            'model': Stopword,
            'serializer': StopwordSerializer
        },
        'synonym': {
            'model': Synonym,
            'serializer': SynonymSerializer
        },
        'representative': {
            'model': Representative,
            'serializer': RepresentativeSerializer
        },
        'edgelist_term': {
            'model': EdgelistTerm,
            'serializer': EdgelistTermSerializer
        },
    }
    return term_model_mapping


if __name__ != '__main__':
    from .models import Terminology, Compound, Stopword, Synonym, Representative, EdgelistTerm
    from terms.serializers import TerminologySerializer, SynonymSerializer, RepresentativeSerializer, StopwordSerializer, CompoundSerializer, EdgelistTermSerializer
    from documents.models import Document
    from analysis.word_extractors.topic_extractor import make_rawdata_class, make_base_dict
    from analysis.word_extractors.noun_extractor import create_noun_compound_df_from_text_list
    from analysis.edgelist.kistep_edgelist import SnaEdgelist
    from terms.userdict_utils import delete_word_after_making_userdict
    from topic_model.topic_utils import TopicModelParser
    from db_manager.managers import FrontDataAccessor
    from db_manager.models import FrontReference
    from config.constants import TARGET_DB

    DOCU_TERM_MAPPER = initialize_term_model()


def create_term_object_from_word_list(target_db, term_type, column_name, word_list, chunk_size=10000,
                                      exclude_stopword=True):
    print(f'{term_type} insert: {target_db}, data_size: {len(word_list)} , chunk_size:{chunk_size}')
    filter_column = f'{column_name}__in'
    filer_query = Q(**{filter_column: word_list})
    target_model = DOCU_TERM_MAPPER[term_type]['model']
    stopword_model = DOCU_TERM_MAPPER['stopword']['model']
    target_db_term_set = set(
        target_model.objects.using(target_db).filter(filer_query).values_list(column_name, flat=True))
    # 불용어 제외
    target_db_stopword_set = set()
    if exclude_stopword:
        target_db_stopword_set = set(
            stopword_model.objects.using(target_db).filter(stopword__in=word_list).values_list('stopword', flat=True))
    insert_target_term = set(word_list) - target_db_term_set.union(target_db_stopword_set)
    res = []
    if len(insert_target_term) == 0:
        return res
    insert_target_term = sorted(insert_target_term)
    insert_target_model = []
    for target_term in insert_target_term:
        t = target_model()
        t.__setattr__(column_name, target_term)
        insert_target_model.append(t)
    res = target_model.objects.using(target_db).bulk_create(insert_target_model, batch_size=chunk_size,
                                                            ignore_conflicts=True)
    print(f'{term_type} insert: {target_db} finished')
    return res


def create_terminology_from_word_list(target_db, word_list, chunk_size=10000, exclude_stopword=True):
    return create_term_object_from_word_list(target_db=target_db, term_type='terminology', column_name='term',
                                             word_list=word_list, chunk_size=chunk_size,
                                             exclude_stopword=exclude_stopword)


def create_stopword_from_word_list(target_db, word_list, chunk_size=10000, exclude_stopword=True):
    return create_term_object_from_word_list(target_db=target_db, term_type='stopword', column_name='stopword',
                                             word_list=word_list, chunk_size=chunk_size,
                                             exclude_stopword=exclude_stopword)


def create_edgelist_term_from_word_list(target_db, word_list, chunk_size=10000, exclude_stopword=False, insert_terminology=True):
    term_res = []
    if insert_terminology:
        term_res = create_term_object_from_word_list(target_db=target_db, term_type='terminology', column_name='term',
                                                    word_list=word_list, chunk_size=chunk_size,
                                                    exclude_stopword=exclude_stopword)
    res = create_term_object_from_word_list(target_db=target_db, term_type='edgelist_term', column_name='term',
                                            word_list=word_list, chunk_size=chunk_size,
                                            exclude_stopword=exclude_stopword)
    # cahce 클리어
    cache.delete(f'{target_db}_edgelist_term')
    return term_res + res


def create_synonym_and_representative_from_word_list(target_db, term_type, main_sub_term_list, chunk_size=10000):
    """

    Parameters
    ----------
    target_db : str, document database
        ex)docu_patent, docu_thesis, etc
    term_type : str, 동의어타입
        ex)synonym, representative
    main_sub_term_list : array-like(array-like), 단어1, 동의어1, 동의어2 의 list로 이루어진 이중리스트
        ex) [['인공지능','artificial intelligence','AI'],['실린더헤드','실린더','헤드'],[...],...]
    chunk_size : int, database insert chunk size
    Returns
    -------
    res : database insert 결과
    error_list : 에러발생 단어 리스트
    """
    main_sub_pair = []
    db_check_target = set()
    error_list = []
    for main_sub_term in main_sub_term_list:
        try:
            main_term = main_sub_term[0]
            sub_terms = main_sub_term[1:]
            db_check_target.add(main_term)
            for sub_term in sub_terms:
                main_sub_pair.append({'main_term': main_term, 'sub_term': sub_term})
                db_check_target.add(sub_term)
        except IndexError:
            error_list.append(main_sub_term)
            continue
    # db에 없는 단어 생성, 동의어와 관련된 단어는 불용어라해서 제외 하지 않음
    # default term 과 중복될 수 있음
    if not db_check_target or len(db_check_target) == 0:
        return [], main_sub_term_list
    main_sub_pair_df = pd.DataFrame(main_sub_pair)
    target_model = DOCU_TERM_MAPPER[term_type]['model']
    main_sub_pair_df['insert_target_model'] = main_sub_pair_df.apply(
        lambda row: target_model(main_term=row['main_term'], sub_term=row['sub_term']), axis=1)
    res = target_model.objects.using(target_db).bulk_create(main_sub_pair_df.insert_target_model.tolist(),
                                                            batch_size=chunk_size, ignore_conflicts=True)
    return res, error_list


def create_compound_from_word_list(target_db, compound_list, chunk_size=10000):
    """

    Parameters
    ----------
    target_db : str, document database
        ex)docu_patent, docu_thesis, etc
    compound_list : array-like(array-like), 복합어, 복합어 구성단어1, 복합어 구성단어n 의 list로 이루어진 이중리스트
        ex) [['제동능력','제동','능력'],['실린더헤드','실린더','헤드'],[...],...]
    Returns
    -------
    res : database insert 결과
    """
    print(f'compound insert: {target_db}')
    compound_only = []
    component_only = []
    for comp in compound_list:
        compound_only.append(comp[0])
        component_only.append(' '.join(comp[1:]))
    compound_df = pd.DataFrame({'compound': compound_only, 'component': component_only})
    target_model = DOCU_TERM_MAPPER['compound']['model']
    # default_compound_model = DOCU_TERM_MAPPER['compound']['model']
    target_db_term_set = set(
        target_model.objects.using(target_db).filter(compound__in=compound_df['compound']).values_list('compound',
                                                                                                       flat=True))
    insert_target_term = set(compound_df['compound']) - target_db_term_set
    compound_df = compound_df[compound_df['compound'].isin(insert_target_term)]
    res = []
    if compound_df.empty:
        return res
    compound_df['insert_target_model'] = compound_df.apply(
        lambda row: Compound(compound=row['compound'], component=row['component']), axis=1)
    # for pos in range(0,len(compound_df),chunk_size):
    #     res += target_model.objects.using(target_db).bulk_create(compound_df[pos:pos+chunk_size].insert_target_model.tolist(), batch_size=chunk_size)
    res = target_model.objects.using(target_db).bulk_create(compound_df.insert_target_model.tolist(),
                                                            batch_size=chunk_size, ignore_conflicts=True)
    print(f'compound insert finished')
    return res


def create_topic_word_from_topic_extractor(target_db):
    we_edge = SnaEdgelist()
    all_edgelist = we_edge.get_all_edgelist(target_db=target_db)
    Edgelist_class = make_rawdata_class(all_edgelist)
    middle_all, topic_all, stop_all = make_base_dict(Edgelist_class, qs=0.001, nm=50)
    edgeterm_res = create_edgelist_term_from_word_list(target_db=target_db, word_list=topic_all, exclude_stopword=True, insert_terminology=False)
    stopword_res = create_stopword_from_word_list(target_db=target_db, word_list=stop_all)
    return edgeterm_res + stopword_res


def create_noun_compound_from_noun_extractor(target_db, insert_db=False, chunk_size=10000):
    """
    target_db database 의 document 테이블의 title과 content을 이용하여 명사 추출
    임시 전처리 함수(명사 추출을 위한 전처리 기능 추가 필요)
    clean_text:
        1. 단어 오른쪽에 위치한 괄호문자, 특정 특수문자(;,") 제거
        2. 한글/숫자와 문자/공백 사이의 . 제거
        3. 단어 왼쪽에 위치한 괄호, 특정 특수 문자 공백 처리
        4. 두개 이상의 공백을 공백 한개로 변경
    Parameters
    ----------
    target_db : str, document database ex) docu_patent, docu_thesis, etc
    insert_db : database 에 명사와 복합어 저장 여부
    Returns 1(if param insert db = False)
    -------------------------------------
    noun_list :  명사단어 리스트
    compound_list : 복합어 단어 리스트
    Returns 2(if param insert_db = True)
    ------------------------------------
    noun_res : database terminology table 에 저장완료된 Terminology Model 리스트
    compound_res : database compound table 에 저장완료된 Compound Model 리스트
    """
    docs = Document.objects.using(target_db).all()
    target_text_set = set()
    noun_list = []
    compound_list = []

    def create_text_list_from_docs_cols(docs, target_cols=None):
        for doc in docs:
            for col in target_cols:
                try:
                    target_text_set.add(doc.__getattribute__(col))
                except AttributeError as e:
                    print(e)
                    continue

    create_text_list_from_docs_cols(docs, ['kor_pjt_nm', 'rsch_goal_abstract', 'exp_efct_abstract', 'rsch_abstract',
                                           'kor_kywd', 'eng_kywd'])
    noun_compound_df = create_noun_compound_df_from_text_list(target_text_set)
    if noun_compound_df.empty:
        return noun_list, compound_list
    # # 임시 저장
    # noun_compound_df.to_parquet(f'/data/temp_parquet/{target_db}_noun_compound_df.parquet')
    # noun_compound_df = pd.read_parquet('/data/temp_parquet/patent_noun_compound_df.parquet')
    compound_df = noun_compound_df[noun_compound_df['noun'].str.contains(' ')].rename({'noun': 'compound'}, axis=1)
    noun_df = noun_compound_df[~noun_compound_df.index.isin(compound_df.index)]
    noun_res = []
    compound_res = []
    if not compound_df.empty:
        compound_list = sorted(compound_df['compound'].str.split(' ').tolist())
        compound_res = create_compound_from_word_list(target_db=target_db, compound_list=compound_list,
                                                      chunk_size=chunk_size)
    if not noun_df.empty:
        # 사용자 사전 추가 전 불용어 제거
        noun_df["delete"] = noun_df.noun.map(delete_word_after_making_userdict)
        noun_df = noun_df.loc[noun_df.delete == False]
        noun_list = sorted(noun_df['noun'].tolist())
        noun_res = create_terminology_from_word_list(target_db=target_db, word_list=noun_list, chunk_size=chunk_size,
                                                     exclude_stopword=True)
    if not insert_db:
        return noun_list, compound_list
    return noun_res, compound_res


def create_noun_compound_from_keyword_col(target_db, insert_db=False, chunk_size=10000):
    """
    target_db database 의 document 테이블의 keyword columns의 값을 사전과 edgelist term에 추가한다.
    Parameters
    ----------
    target_db : str, document database ex) docu_patent, docu_thesis, etc
    insert_db : database 에 명사와 복합어, 엣지리스트텀 저장 여부
    Returns 1(if param insert db = False)
    -------------------------------------
    noun_list :  명사단어 리스트
    compound_list : 복합어 단어 리스트
    Returns 2(if param insert_db = True)
    ------------------------------------
    noun_res : database terminology table 에 저장완료된 Terminology Model 리스트
    compound_res : database compound table 에 저장완료된 Compound Model 리스트
    """
    target_cols = ['kor_kywd']
    docs = Document.objects.using(target_db).values('kor_kywd')
    keyword_set = set()
    noun_list = []
    compound_list = []

    # db 불러와서 전처리
    for doc in docs:
        for col in target_cols:
            try:
                if isinstance(doc[col],str):
                    keyword_set.update(doc[col].split(","))

            except AttributeError as e:
                print(e)
                continue

    noun_compound_df = pd.DataFrame(list(keyword_set),columns=['noun'])
    if noun_compound_df.empty:
        return noun_list, compound_list

    # 특수문자 포함, 한글 없는 것 대상에서 제외
    only_word = re.compile(r"^[a-zA-Z가-힣\s]+$")
    space_englist_filter = re.compile(r"^[a-zA-Z\s]+$")
    noun_compound_df = noun_compound_df.loc[noun_compound_df.noun.map(lambda x: True if only_word.search(x) else False)]
    noun_compound_df = noun_compound_df.loc[noun_compound_df.noun.map(lambda x: False if space_englist_filter.search(x) else True)].reset_index(drop=True)
    noun_compound_df['noun'] = noun_compound_df.noun.str.lower().str.strip()

    compound_df = noun_compound_df[noun_compound_df['noun'].str.contains(' ')].rename({'noun': 'component'}, axis=1)
    noun_df = noun_compound_df[~noun_compound_df.index.isin(compound_df.index)]

    # 전처리 및 db 추가
    noun_res = []
    compound_res = []
    compound_list = []
    if not compound_df.empty:
        compound_df['compound_set'] = compound_df.component.map(lambda x: x.replace(" ","") + ' ' + x)
        compound_df = compound_df.loc[compound_df.component.str.replace(" ", "").str.len() <= 15]
        compound_df = compound_df.drop_duplicates("compound_set")
        compound_set_list = sorted(compound_df['compound_set'].str.split(' ').tolist())
        compound_list = [com_list[0] for com_list in compound_set_list]
        compound_res = create_compound_from_word_list(target_db=target_db, compound_list=compound_set_list,
                                                      chunk_size=chunk_size)
        compound_edge_res = create_edgelist_term_from_word_list(target_db=target_db, word_list=compound_list, chunk_size=10000,
                                                       exclude_stopword=False, insert_terminology=False)
    if not noun_df.empty:
        # 사용자 사전 추가 전 불용어 제거
        noun_df["delete"] = noun_df.noun.map(delete_word_after_making_userdict)
        noun_df = noun_df.loc[(noun_df.delete == False) & (noun_df.noun.str.len() <= 15)]
        noun_list = sorted(noun_df['noun'].tolist())
        # 복합어 말고 단어는 edgelist랑 termilogy 함께 넣어주기
        noun_res = create_edgelist_term_from_word_list(target_db=target_db, word_list=noun_list, chunk_size=10000, exclude_stopword=False, insert_terminology=True)

    if not insert_db:
        return noun_list, compound_list
    return noun_res, compound_res


def create_topic_synonym_list_from_topicmodel(target_db, term_type='synonym', insert_db=True):
    """
    create_date: "2022-01-13"

    """
    past_modf_dt = ''
    # 저장된 토픽모델 last_modf_dt 값 불러오기
    last_topic_model_modf_dt = FrontReference.objects.using(target_db).filter(data_key='last_topic_model_modf_dt').values_list('data_value',flat=True)
    if last_topic_model_modf_dt:
        past_modf_dt = last_topic_model_modf_dt[0]
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    dt_condition = f'del_yn = "N" AND last_modf_dt <= "{current_time}"'
    if past_modf_dt != '':
        dt_condition += f' AND last_modf_dt > "{past_modf_dt}"'
    fda = FrontDataAccessor(target_db=target_db)
    # front db topic_model 테이블에서 현재시간보다는 작고, 마지막으로 불러온 데이터의 수정시간보다는 큰 데이터 가져오기
    topic_model_df = fda.get_front_data_df(table_name='topic_model', required_cols=['topic_id','topic_model','last_modf_dt'], where_condtion=dt_condition)
    if topic_model_df.empty:
        return []
    new_past_modf_dt = topic_model_df.last_modf_dt.max()
    target_db_dict = {}
    for topic_id, topic_model in topic_model_df[['topic_id', 'topic_model']].to_dict(orient='split')['data']:
        try:
            if not TopicModelParser(topic_model).figure_relation_df().empty:
                new_topic_dict = {topic_id: {"topic_model": topic_model, "rel_word_df": TopicModelParser(topic_model).figure_relation_df()}}
                target_db_dict.update(new_topic_dict)
        except ValueError as e:
            print(f'_error_:{topic_id} {e}')
            continue

    topic_words_df = pd.DataFrame(columns=["main_term", "sub_term", 'relation'])

    for topic_id, topics in target_db_dict.items():
        topic_words_df = pd.concat([topic_words_df, topics['rel_word_df']])
    topic_words_df = topic_words_df.loc[topic_words_df.main_term != topic_words_df.sub_term]
    topic_words_df = topic_words_df.drop_duplicates()

    topic_rel_words_df = topic_words_df[["main_term", "sub_term"]].groupby('main_term')['sub_term'].agg(
        ",".join).reset_index()
    topic_rel_words_df['synonym'] = topic_rel_words_df['main_term'] + ',' + topic_rel_words_df['sub_term']
    topic_rel_words_df['synonym_list'] = topic_rel_words_df['synonym'].map(lambda x: x.split(','))
    topic_synonym_list = topic_rel_words_df.synonym_list.tolist()

    if insert_db:
        synonyms_res, _ = create_synonym_and_representative_from_word_list(target_db=target_db, term_type=term_type, main_sub_term_list=topic_synonym_list, chunk_size=10000)
        # 토픽모델 last_modf_dt 값으로 기준점 업데이트
        FrontReference.objects.using(target_db).update_or_create(data_key='last_topic_model_modf_dt',defaults={'data_value':new_past_modf_dt})
        return synonyms_res
    return topic_synonym_list


def validate_ids(id_list):
    valid_list = []
    error_list = []
    for id in id_list:
        try:
            id = int(id)
            valid_list.append(id)
        except ValueError:
            error_list.append(id)
            continue
    return valid_list, error_list


def delete_all_data(target_db, target_table):
    print(target_db, target_table, 'delete all data')
    target_model = DOCU_TERM_MAPPER[target_table]['model'].objects.using(target_db).all().delete()


def validate_compound(compound_component_list):
    valid_list = []
    error_list = []
    for comp_set in compound_component_list:
        if not isinstance(comp_set, Iterable) or len(comp_set) < 2:
            error_list.append(comp_set)
            continue
        compound = comp_set[0]
        component = comp_set[1:]
        if compound != "".join(component):
            error_list.append(comp_set)
            continue
        valid_list.append(comp_set)
    return valid_list, error_list


def validate_sub_term_exists(target_term_list):
    error_list = []
    clean_list = []
    for term_list in target_term_list:
        if not isinstance(term_list, list) or len(term_list) < 2:
            error_list.append(term_list)
        else:
            clean_list.append(term_list)
    return clean_list, error_list


def validate_unhashable_type(target_term_list):
    error_list = []
    clean_list = []
    for term_list in target_term_list:
        if not isinstance(term_list, str):
            error_list.append(term_list)
        else:
            clean_list.append(term_list)
    return clean_list, error_list


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
    from analysis.word_extractors.topic_extractor import make_rawdata_class, make_base_dict
    from analysis.word_extractors.noun_extractor import create_noun_compound_df_from_text_list
    from analysis.edgelist.kistep_edgelist import SnaEdgelist
    import timeit
    from terms.models import Terminology, Compound, Stopword, Synonym, Representative, EdgelistTerm
    from terms.serializers import TerminologySerializer, SynonymSerializer, RepresentativeSerializer, \
        StopwordSerializer, CompoundSerializer, EdgelistTermSerializer
    from documents.models import Document
    from terms.userdict_utils import delete_word_after_making_userdict
    from topic_model.topic_utils import TopicModelParser
    from db_manager.managers import FrontDataAccessor
    from db_manager.models import FrontReference
    from config.constants import TARGET_DB

    create_topic_synonym_list_from_topicmodel(target_db='kistep_sbjt')


    DOCU_TERM_MAPPER = initialize_term_model()
    start_time = timeit.default_timer()
    # target_db = 'docu_test'
    # job_target = ['limenet_analysis']
    # for target_db in job_target:
    #     create_noun_compound_from_keyword_col(target_db, insert_db=False, chunk_size=10000)
    # print(timeit.default_timer() - start_time)
    # word_list = ['인공지능','게놈프로젝트','인천상륙작전','인센티브','리눅스','로컬라이징']
    # res = create_terminology_from_word_list('docu_test',word_list)
    # res = create_topic_word_from_topic_extractor(target_db)
    # print(res)
    # delete
    # for target_db in job_target:
    #     for table_name in ['compound','terminology']:
    #         delete_all_data(target_db=target_db, target_table=table_name)