from gensim.corpora import dictionary
from numpy.core.fromnumeric import resize
import pandas as pd
import numpy as np
from itertools import chain, repeat, compress, combinations, product
import itertools
import gensim
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
import json
import os, warnings
from multiprocessing import Pool, cpu_count, Process, Manager


# from sna_functions import make_one_mode_graph
if __name__ != '__main__':
    from .sna_functions import make_one_mode_graph
    from config.constants import SUPPORT_ANALYSIS_TYPES, TARGET_DB, ES_DOC_SOURCE

warnings.filterwarnings('ignore')
"""
* nodes
id : 노드의 고유번호 | int
name : 노드의 이름 | chr
type : 노드의 유형(검색어 or 토픽 or 단어) | chr

* edges
source : source | int
target : target | int
check : 단어 포함 여부 defalut 값(Y/N) | chr
relation : 노드간 관계 속성 (R: 관련어, 'S':동의어)| str
# search_text - topic 속성 노드 끼리의 연결에는 check에 None값이 들어감

# example
{'nodes': [{
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
    'type' : 'word
   }
}],
 'links': [{'source': 1, 'target': 2, 'check': 'Y'},
  {'source': 1, 'target': 3, 'check': 'N'}]
}

"""

def make_lda_data(_edgelist, remove_n, no_below, no_above):

    numbers = [len(p) for p in _edgelist['position']]
    words = list(chain.from_iterable((repeat(_word, n) for (_word, n) in zip(_edgelist['word'], numbers))))
    papers = list(chain.from_iterable((repeat(_paper, n) for (_paper, n) in zip(_edgelist['doc_id'], numbers))))
    positions = list(chain(*_edgelist['position']))

    token_doc = pd.DataFrame({'_doc': papers, 
                              'token': words, 
                              'pos': positions}).sort_values(['_doc', 'pos'])

    token_doc = token_doc.groupby(token_doc['_doc']).apply(lambda x: ','.join(x['token']))

    token_list = [d.split(',') for d in token_doc]
    token_list = pd.Series(token_list)  # 토큰화
    
    """
    단어를 id로 바꾸고 뜻을 dictionary로 만들기
    각 단어를 (단어 id, 나온 횟수)로 바꾸는 작업

    - corpus[i]: i번째 문서에서 나온 단어들을 가지고 (단어 id, 나온 횟수)들을 저장한 list
    - dictionary[i]: id가 i인 단어

    """
    dictionary = gensim.corpora.Dictionary(token_list)
    dictionary.filter_n_most_frequent(remove_n)
    dictionary.filter_extremes(no_below = no_below, no_above = no_above)

    corpus = [dictionary.doc2bow(text) for text in token_list]

    return corpus, token_list, dictionary

def get_coherence(_model, corpus, token_list, coherence):    

    cm = CoherenceModel(model=_model, corpus=corpus, texts=token_list, coherence=coherence, window_size=15, topn=10)
    coherence = cm.get_coherence()
    
    return coherence

def make_lda_dataframe(_model, optimal_num, dictionary, p_limit):    # p_limit = 0.1

    def lda_add_index(_model, k):
        _df = pd.DataFrame(_model.get_topic_terms(k, 10), columns = ['word', 'attr'])
        _df.index = [k] * len(_df)

        return _df

    _lda = [lda_add_index(_model, k) for k in range(0, optimal_num)]   
    _lda = pd.concat(_lda)

    _lda = _lda.reset_index()
    _lda = _lda.rename(columns = {'index': 'topic'})
    _lda['word'] = [dictionary[w] for w in _lda['word']]

    # _lda = _lda.groupby('word').apply(lambda x: x.sort_values('attr').iloc[0])
    _lda = _lda.sort_values(['topic', 'attr'], ascending = [True, False]).reset_index(drop = True)
    # _lda = _lda.groupby('topic').apply(lambda x: x.iloc[0:10]).reset_index(drop=True) # max print : 10
    
    if len(_lda.topic.unique()) > len(_lda.word.unique()):
        _lda = pd.DataFrame(columns = ['topic', 'word', 'attr'])

        return _lda

    _topic = _lda.groupby('topic').apply(lambda x: ','.join(x.word.iloc[0:10]).split(','))
    _topic_nm = [t[0] for t in _topic]

    n = 1

    while len(_topic_nm) != len(set(_topic_nm)):
        _idx = [idx for idx, item in enumerate(_topic_nm) if item in _topic_nm[:idx]]
        _topic_dup = [t[n] for t in _topic]
        
        for i in _idx:
            _topic_nm[i] = _topic_dup[i]

        n += 1
        
        if n > 10:
            print('All duplicates')

        if len(_topic_nm) == len(set(_topic_nm)):
            break

    _topic_nm = pd.Series(_topic_nm, index = list(range(0, optimal_num))).rename('topic_nm')
    _lda = pd.merge(_lda, _topic_nm, left_on = 'topic', right_index = True)
    _lda = _lda.drop('topic', axis = 1).rename(columns = {'topic_nm': 'topic'})
    _lda = _lda.iloc[:, [2, 0, 1]]

    _lda['c_check'] = np.round(_lda.groupby('topic').apply(lambda x: x['attr'].cumsum()).tolist(), 1)
    _topics = _lda.groupby('topic').apply(lambda x: any(x['c_check'] >= p_limit))
    _topics = list(compress(_topics.index.tolist(), _topics))
    _lda = _lda[_lda['topic'].isin(_topics)]
    _lda = _lda.reset_index(drop = True)

    t_start = _lda.groupby('topic').apply(lambda x: x.iloc[0].name)
    t_end = _lda.groupby('topic').apply(lambda x: x[x['c_check'] >= p_limit].iloc[0].name)

    t_list = [list(range(t_start[n], t_end[n] + 1)) for n in range(0, len(t_start))]
    t_list = list(chain(*t_list))

    _lda = _lda.iloc[t_list].reset_index(drop = True)

    # 선택 여부 비활성화
    # _lda['check'] = ['Y' if attr >= 0.01 else 'N' for attr in _lda.attr]
    _lda['check'] = 'N'
    _lda = _lda.drop(['attr', 'c_check'], axis = 1)    

    return _lda

def make_json(node_data, edge_data):
    # input : dataframe
    # output : json
    json_node = node_data.to_json(orient="records")
    json_link = edge_data.to_json(orient="records")
    json_node = json.loads(json_node)
    json_link = json.loads(json_link)

    made_json = {
        "nodes": json_node,
        "links": json_link
    }

    return made_json

def name_adequacy_check(_lda, _edgelist):
    """
    lda에서 topic 이름을 정할 때, 해당 topic에 속할 확률이 가장 높은 단어를 topic 이름으로 선정하는 것보다
    연관성 분석에서 사용하는 지표 중 하나인 lift 값을 이용해서 해당 값이 가장 높은 단어를 topic 이름으로 선정.

    """

    if _lda.empty:
        return _lda

    _lda_tn = _lda.topic.unique()
    _lda_a = []    # adequate
    _lda_na = []    # not adequate

    for t in _lda_tn:
        _t = _lda.groupby('topic').get_group(t)

        if len(_t) == 1:
            _lda_a.append(_t)

        else:
            _lda_na.append(_t)
    
    if not _lda_na:
        _lda_filter_df = pd.concat(_lda_a)

    else:
        _lda = pd.concat(_lda_na)
        _topic = _lda.groupby('topic').apply(lambda x: list(combinations(x.word.tolist(), 2)))

        bag_word_list = []
        doc_list = _edgelist.groupby('doc_id').apply(lambda x: ','.join(x.word).split(','))

        for t in _topic:
            bag_word = []

            for w1, w2 in t:
                for d in doc_list:
                    t_1 = d.count(w1)
                    t_2 = d.count(w2)

                    if (t_1 == 1) & (t_2 == 1):
                        t_3 = 1

                    else:
                        t_3 = 0
                
                    bag_word.append([t_1, t_2, t_3, w1 + ',' + w2])
            
            bag_word_list.append(bag_word)

        def get_other_topic_name(bag_word_list, t_n):

            bag_word = pd.DataFrame(bag_word_list[t_n], columns = ['w_1', 'w_2', 'w_12', 'word'])

            bag_word = bag_word.groupby('word')
            bag_word = [bag_word.get_group(','.join(r)) for r in _topic[t_n]]

            bag_word_result = [(b.iloc[:, [0, 1, 2]].sum(axis = 0) / len(b)).tolist() for b in bag_word]

            _li = [w3 / (w1 * w2) for w1, w2, w3 in bag_word_result]

            _lift = pd.DataFrame(_topic[t_n], columns = ['source', 'target'])
            _lift['weight'] = _li

            # _lift = _lift[_lift.weight != 0]
            g = make_one_mode_graph(_lift)

            g_result = pd.DataFrame({'word': g.vs['name'],
                                     'w_degree': g.strength(weights = g.es['weight']),
                                     'eigen': g.eigenvector_centrality(),
                                     'close': g.closeness()})

            g_result = g_result.sort_values('w_degree', ascending = False)
            g_result = g_result.reset_index(drop = True)

            return g_result

        _lda_t = _lda.groupby('topic')
        t_dict = {i:v for i, v in enumerate(_lda.topic.unique())}
        other_df = []

        for i in range(len(t_dict)):
            o_df = _lda_t.get_group(t_dict[i])
            o_word = get_other_topic_name(bag_word_list, i)['word']

            emp_df = pd.DataFrame(columns = ['topic', 'word'])
            emp_df['word'] = o_word
            emp_df['topic'] = o_word[0]

            if t_dict[i] != o_word[0]:
                o_df = pd.merge(emp_df, o_df.drop('topic', axis = 1), on = ['word'], how = 'left')

            else:
                pass

            other_df.append(o_df)
            
        _lda_filter_df = pd.concat(_lda_a + other_df).reset_index(drop=True)

    return _lda_filter_df


def lda_fullprocess(search_text, _edgelist, s_dict, remove_n, no_below, no_above, _start, _end, p_limit):
    
    # _edgelist = _edgelist[_edgelist.col_type == 'content']

    #####################################################
    if not s_dict:
        s_dict = {i:0 for i in _edgelist.doc_id.unique()} 
    
    _edgelist_doc = _edgelist.copy()
    select_doc = list(itertools.islice(s_dict.keys(), 300))
    _edgelist = _edgelist_doc[_edgelist_doc.doc_id.isin(select_doc)]
    #####################################################
    
    if _edgelist.empty:
        _node = pd.DataFrame(columns = ['id', 'name', 'type'])
        _edge = pd.DataFrame(columns = ['source', 'target'])

        A = make_json(_node, _edge)

        return A
    
    corpus, token_list, dictionary = make_lda_data(_edgelist, remove_n, no_below, no_above)
    
    if not list(chain(*corpus)):
        _node = pd.DataFrame(columns = ['id', 'name', 'type'])
        _edge = pd.DataFrame(columns = ['source', 'target'])

        A = make_json(_node, _edge)

        return A

    _k = [i for i in range(_start, _end + 1)]
    
    # for문 반복 최적화 1
    multi_manager = Manager()
    multi_model_list = multi_manager.list()
    process_list = []

    def get_lda_model(k):
        """
        num_topics: topic 갯수
        passes: 알고리즘 동작 횟수(epoch와 비슷한 듯)
        num_words: 각 topic별 출력할 단어
        alpha, eta: 디리클레 분포의 감마함수에 대한 파라미터(alpha, beta)
    
        """
        lda_model = LdaModel(corpus, num_topics = k, id2word = dictionary, passes = 5, iterations = 300, random_state = 2021, alpha = 'auto', eta = 'auto')
        multi_model_list.append((lda_model, k))

    for k in _k:    # topic 갯수를 조정해 가면서 lda 모델 생성 
        p = Process(target = get_lda_model, args = (k, ))
        process_list.append(p)
    for process in process_list:
        process.start()
    for process in process_list:
        process.join()

    model_list = list(multi_model_list)
    
    # multiprocess 결과로 나오는 list는 작업 완료 순서에 따라 다르기 때문에 순서가 섞임(k=2, 10까지 순차적으로 생성되지 않음)
    # 모델 생성 시 사용한 k값 기준으로 sorting
    model_i = [m[1] for m in model_list]
    model_list = [m[0] for _, m in sorted(zip(model_i, model_list))]
    # model_list = [LdaModel(corpus, num_topics=k, id2word=dictionary, passes=5, iteratio 

    """
    coherence는 주제의 일관성을 측정한다.
        - topic이 얼마나 의미론적으로 일관성 있는지
        - 높을수록 의미론적 일관성 높음
        - 해당 모델이 얼마나 실제로 의미있는 결과를 내는지 확인하기 위해 사용

    """
    cohe_list = np.array([get_coherence(m, corpus, token_list, 'c_v') for m in model_list])    # c_v, c_uci
    optimal_num = _k[cohe_list.argmax()]
    _model = model_list[optimal_num - _start]    # _model: cohe score가 가장 높은 lda model
    
    _lda = make_lda_dataframe(_model, optimal_num, dictionary, p_limit)
    _lda = name_adequacy_check(_lda, _edgelist)

     # test
    topics = _model.print_topics(num_words = 4)

    for i, topic_list in enumerate(_model[corpus]):
        if i == 5:
            break
        print(i, '번째 문서의 topic 비율은 ', topic_list)

    _lda = _lda.reset_index(drop = True)
    _lda['word'] = [w + ',' + t for w, t in zip(_lda['word'], _lda['topic'])]
    
    if _lda.empty:
        _node = pd.DataFrame(columns = ['id', 'name', 'type'])
        _edge = pd.DataFrame(columns = ['source', 'target'])

        A = make_json(_node, _edge)
        
        return A

    _node = [search_text] + _lda.topic.unique().tolist() + _lda.word.unique().tolist()
    _node = pd.DataFrame({'name': _node}).reset_index().rename(columns = {'index': 'id'})
    _node['type'] = ['search_text'] + ['topic'] * _lda.topic.unique().size + ['word'] * (len(_node) - (_lda.topic.unique().size + 1))
    _node.name = [n + ',' + t if t != 'word' else n for t, n in zip(_node.type, _node.name)]

    _lda.topic = [l + ',topic' for l in _lda.topic]
    _edge = _lda.rename(columns = {'topic': 'source', 'word': 'target'})
    _lda.topic = [t.split(',')[0] for t in _lda.topic]

    items = [[search_text + ',search_text'], [l + ',topic' for l in _lda.topic.unique().tolist()]]
    h_edge = pd.DataFrame.from_records(list(product(*items)), columns = ['source', 'target'])
    h_edge['check'] = 'N'
    _edge = pd.concat([h_edge, _edge])

    d_node = {v:i for i,v in enumerate(_node.name)}
    _edge.source = _edge.source.map({k:v for k, v in zip(d_node.keys(), d_node.values())})
    _edge.target = _edge.target.map({k:v for k, v in zip(d_node.keys(), d_node.values())})
    _edge['relation'] = 'R'
    _edge = _edge.drop_duplicates()

    _node.name = [n.split(',')[0] for n in _node.name]

    A = make_json(_node, _edge)

    return A


if __name__ =='__main__':
    # 파일 직접 실행 시 필요한 setup 부분
    import sys
    import os
    from pathlib import Path
    APP_DIR = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(APP_DIR))
    os.environ['DJANGO_SETTINGS_MODULE'] = 'config.settings.base'
    os.environ['DJANGO_ALLOW_ASYNC_UNSAFE'] = 'true'
    import django
    django.setup()
    
    from analysis.edgelist.kistep_edgelist import SnaEdgelist, PlayerEdgelist
    from analysis.word_extractors.topic_extractor import *
    from sna_functions import *
    from documents.document_utils import create_topic_model_search_query, create_keyword_search_query
    from db_manager.managers import ElasticQuery
    import timeit
    from config.constants import SUPPORT_ANALYSIS_TYPES, TARGET_DB, ES_DOC_SOURCE

    start_time = timeit.default_timer()

    # document > view > analysis
    # 3 ~ 15
    # 3 ~ 7
    # 4 ~ 7
    # 자율주행, 로봇
    
    ## 자율주행 : 348개
    # 기존 : 40초
    # 3~7 : 16초    

    target_db = 'kistep_sbjt'
    we_edge = SnaEdgelist()
    p_edge = PlayerEdgelist()
    
    es_query = ElasticQuery(target_db)
    search_form = {
        'search_text':['인공지능']    # 로봇
    }

    def get_search_result_by_full_query(query, filter_percent=10):
        es_query = ElasticQuery(target_db=target_db)
        min_score = es_query.get_percentile_score(query['query'],filter_percent=filter_percent)
        query.update({'min_score': min_score})
        search_result = es_query.scroll_docs_by_query(query=query['query'], source=ES_DOC_SOURCE, min_score=min_score)
        return search_result
    
    query = create_keyword_search_query(search_form=search_form)
    # search_result = es_query.get_docs_by_full_query(query=query, doc_size=doc_size)
    search_result = get_search_result_by_full_query(query=query, filter_percent=10)
    doc_ids_score = {x['_id']: 0 if x['_score'] is None else x['_score'] for x in search_result['hits']['hits']}
    doc_ids = list(doc_ids_score.keys())
    doc_size = len(doc_ids)
    print(doc_size)
    # _doc = pd.read_parquet('./compound_df.parquet')
    # _ids = _doc.doc_id.unique().tolist()
    # _size = len(_ids)
    # # all_edgelist = we_edge.get_all_edgelist(target_db=target_db)
    
    ego_edgelist, resize_ego_edgelist = we_edge.get_sna_refined_ego_edgelist(target_db=target_db, doc_ids=doc_ids, size=doc_size, extract_compound=False)  

    A = lda_fullprocess(search_text = search_form['search_text'][0], _edgelist = resize_ego_edgelist, s_dict = doc_ids_score, remove_n = 2, no_below = 2, no_above = 1, _start = 2, _end = 10, p_limit = 0.1)
    print(A)

    print(timeit.default_timer() - start_time)