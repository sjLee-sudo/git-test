if __name__ !='__main__':
    from analysis.edgelist.kistep_edgelist import SnaEdgelist
    from analysis.word_extractors.topic_extractor import *
    from .sna_functions import *
    from .tree_functions import *

import sys
import os
import timeit
import json
from itertools import product

"""
# json 구조

* nodes
id : 노드의 고유번호 | int
index_key : 해당 노드의 유형 | chr
name : 노드의 이름 | chr
depth : 트리 단계 구분 | chr
link_size : 연결된 엣지의 수 | int
pyear : 문서가 작성된 연도 | chr

# pyear는 문서 노드에만 들어가며, 그 외 노드에는 pyear값이 비어있음

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
{'nodes' : [{
    'id': 'db_parentP1'
    'index_key': '10002012',
    'name' : '인공지능연구',
    'depth' : '3-4',
    'link_size' : 14,
    'pyear' : '2018'
}],
 'edges' : [{
     'source' : 'db_parentSA',
     'target' : 'db_parentP1',
     'rel' : '3-section-paper',
     'weight' : 1
 }],
  'outer_links' : [{
      'source':'db_parentP1', 'target':'db_parentP12', 'v':1
  }],
  'search_text' : '인공지능'
}
  
"""
filtering_dict = {'cdfreq': 1, 'dfreq': 1, 'node_limit_base': 1, 'root_parent_base': 400,
                't_cdfreq': 1, 'term1_limit': 100, 'term_ratio': 0.1, 'weight_filter': 1}
    
def extract_topic_word(EgoEdgelist, ResizeEdgelist, window_size=15, alpha=0.0):

    if (EgoEdgelist.empty) & (ResizeEdgelist.empty):
        return None, None, None
    
    EgoEdgelist = EgoEdgelist[EgoEdgelist.col_type == 'content']
    # EgoEdgelist = EgoEdgelist.drop(['bm25', 'dfreq'],axis=1)
    # 본문은 없고 제목만 있는 경우 (임시 처리)
    if EgoEdgelist.empty:
        return None, None, None
    
    EgoEdgelist = EgoEdgelist.reset_index(drop=True)
    dfreq = EgoEdgelist.groupby('word').size().rename('dfreq') # 검색된 문서 내 dfreq를 새롭게 계산
    EgoEdgelist = pd.merge(EgoEdgelist, dfreq, left_on = 'word', right_index = True)
    EgoEdgelist = EgoEdgelist[EgoEdgelist.dfreq > 2] # 최소 2개 이상의 문서에 등장하는 단어만 사용하도록 조정, 1개 문서 내에만 존재하는 강한 연결을 만들지 않도록 함
    
    if EgoEdgelist.empty:
        return None, None, None
    
    pmi_mat, p_rc, w_dict = make_pmi(EgoEdgelist, window_size=window_size, alpha=alpha, sample_n = False, min_pmi = 0)
    # w_dict_inv = {v:i for i, v in w_dict.items()}
    p_rc = p_rc.tocsr()
    
    p_rc.data = -np.log(p_rc.data)
    pmi_mat_s = sparse.dok_matrix(pmi_mat.shape) 
    pmi_mat_s[p_rc.nonzero()] = pmi_mat[p_rc.nonzero()] / p_rc[p_rc.nonzero()] # pmi 행렬 표준화, pmi값을 ln(p_rc)로 나눈 값이다. (ln : 자연로그)
    pmi_mat = pmi_mat_s.tocsr()
    
    topic_word, word_json = word_network(ResizeEdgelist, pmi_mat, w_dict)
    paper_json = paper_network(ResizeEdgelist, EgoEdgelist, topic_word, _revert = 1)
    return topic_word, word_json, paper_json       

def paper_network(ResizeEdgelist, EgoEdgelist, topic_word, _revert = 1):

    resize_data = copy.copy(ResizeEdgelist)

    if not topic_word:
        pass
    else:
        resize_data = resize_data[resize_data.word.isin(topic_word)]
    
    # ego_edgelist
    ego_data = copy.copy(EgoEdgelist)
    # N = ego_data.doc_id.unique().size
    # ego_data['bm25'] = bm25(N,ego_data.dtfreq, ego_data.dfreq).rename('bm25')
    # term_indice_df = ego_edgelist_node_controll(ego_data, g, filtering_dict)
    # ego_data = complete_egoedgelist(ego_data, term_indice_df, filtering_dict)
    e_raw, n_raw, g_raw = df2edge(ego_data, 'paper')

    # resize_ego_edgelist
    dfreq = resize_data.groupby(['word']).size().rename('dfreq')
    resize_d_edgelist = pd.merge(resize_data, dfreq, left_on = 'word', right_index = True)
    g = df2edge(resize_data, 'paper')[2]

    N = resize_d_edgelist.doc_id.unique().size
    resize_d_edgelist['bm25'] = bm25(N,resize_d_edgelist.dtfreq, resize_d_edgelist.dfreq).rename('bm25')
    term_indice_df = ego_edgelist_node_controll(resize_d_edgelist, g, filtering_dict)
    resize_d_edgelist = complete_egoedgelist(resize_d_edgelist, term_indice_df, filtering_dict)

    e,n,g = df2edge(resize_d_edgelist, 'paper')

    # reverting
    n_cluster = n.loc[:,['name','cluster']].set_index('name')
    center_list = [n[n.cluster == c].sort_values('linklen', ascending = False).name.iloc[0:_revert].tolist() for c in range(1,max(n.cluster))]
    center_list = list(itertools.chain(*center_list))

    e_revert = pd.merge(e_raw,n_cluster, left_on = 'source', right_index = True)
    e_revert = pd.merge(e_revert,n_cluster, left_on = 'target', right_index = True)
    e_revert = e_revert[(e_revert.source.isin(center_list)) | (e_revert.target.isin(center_list))]

    e_revert = e_revert[(e_revert.cluster_x == 0) | (e_revert.cluster_y == 0)].drop_duplicates(['cluster_x','cluster_y'])
    e_revert = e_revert.drop(['cluster_x','cluster_y'], axis=1)
    e = pd.concat([e,e_revert]).reset_index(drop=True)

    ## 노드 이름 변경

    g = make_one_mode_graph(e)
    E = g.eigenvector_centrality()
    E = pd.DataFrame({'sE':E},index = g.vs['name'])
    n = pd.merge(n,E,left_on = 'name', right_index = True)

    e = e.rename(columns = {'weight':'v'})
    n = n.drop(columns = ['cluster', 'linklen'])
    n = n.rename(columns = {'name':'t'})
    
    _json = make_json(n, e)
    
    return _json

def word_network(ResizeEdgelist, pmi_mat, w_dict, min_P=0.0, N=300):
    
    resize_data = copy.copy(ResizeEdgelist)
    resize_data = resize_data[resize_data.col_type == 'content']
    e,n,g = df2edge(resize_data, 'word')
    
    P = []
    for r,c,w in e.values:
        try:
            P.append(pmi_mat[w_dict[r], w_dict[c]])
        except:
            P.append(0)
    P = np.array(P, dtype=float)
    
    # P = np.array([pmi_mat[w_dict[r],w_dict[c]] for r,c,w in e.values]) # source, target 사이의 PMI값 확인
    e_reduce = e.iloc[np.nonzero(P>min_P)[0]] # extract_topic_word에서 계산된 표준화 PMI행렬 사용, 0~1사이의 값이 있으며 1에 가까울수록 강한 연결
    g = make_one_mode_graph(e_reduce)

    E = np.array(g.eigenvector_centrality())
    T = [g.vs['name'][i] for i in E.argsort()[::-1]][0:N]
    
    #### 추후 삭제 가능 부분
    
    n_E = pd.DataFrame({'name':g.vs['name'],'sE':E})
    n = pd.merge(n, n_E, on = 'name') 
    ####
    
    e_reduce = e_reduce[e_reduce.source.isin(T)]
    e_reduce = e_reduce[e_reduce.target.isin(T)]
    n_reduce = n[n.name.isin(T)]
    
    #### 추후 삭제 가능 부분
    e_reduce = e_reduce.rename(columns = {'weight':'v'})
    
    n_reduce = n_reduce.drop(columns = ['cluster','linklen'])    
    n_reduce = n_reduce.rename(columns = {'name':'t'})    
    ####
    
    _json = make_json(n_reduce, e_reduce)
    
    return T, _json

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

def treedata_maker(search_text, target_db, is_query_string, size=100):
    we_edge = SnaEdgelist()
    _data, resize_ego_edgelist = we_edge.get_sna_refined_ego_edgelist(target_db=target_db,search_text=search_text, is_query_string=is_query_string, size=size)

    if _data.empty or resize_ego_edgelist.empty:
        emp = pd.DataFrame(columns=['doc_id', 'word', 'col_type', 'dtfreq', 'position', 'section','d_class', 'title'])
        return emp, None, None, target_db, None
    
    topic_word, word_json, paper_json = extract_topic_word(_data, resize_ego_edgelist)   
    p_dict = {i:v for i,v in enumerate(_data.doc_id.unique())}

    if not topic_word:
        pass
    else:
        _data = _data[(_data.word.isin(topic_word)) & (_data.col_type == 'content')]

    w_dict = {i:v for i,v in enumerate(_data.word.unique())}

    return _data, w_dict, p_dict, target_db, paper_json

def tree_sna_main(search_text, target_db_list, is_query_string=False, size=100):    
    
    # step1. db별로 _data 생성
    
    tree_list = [treedata_maker(search_text, t, is_query_string, size) for t in target_db_list]    

    # step2. depth 1-2까지 tree구조 생성
    
    tree_node = [node_maker(_df, _w, _p, _db) for _df, _w, _p, _db, _json in tree_list]
    tree_edge = [edge_maker(_df, _w, _p, _db) for _df, _w, _p, _db, _json in tree_list]
    outer_links = [outlink_maker(_json, _p, _db) for _df, _w, _p, _db, _json in tree_list]
    
    _node = pd.concat(tree_node)
    _edge = pd.concat(tree_edge)

    # step3. step2의 결과들을 하나로 합치고, depth 0-1과 search_text, outer_links를 추가하여 json 생성
    
    head_node = pd.DataFrame({'id':'RT01', 'index_key':'search_text', 'name':search_text, 'depth':'0-1', 'link_size':0}, index=[0])
    
    items = [['RT01'], target_db_list]
    head_edge = pd.DataFrame.from_records(list(product(*items)), columns = ['source','target'])
    head_edge['rel'] = '1-root-index'
    head_edge['weight'] = 1
    head_edge['section'] = ''

    _node = pd.concat([head_node, _node]).sort_values('depth')
    _edge = pd.concat([head_edge, _edge])
    _node = _node.fillna('')
    A = make_json(_node, _edge)

    outer_links = pd.concat(outer_links)
    outer_links = outer_links.to_json(orient='records')
    outer_links = json.loads(outer_links)

    A['outer_links'] = outer_links
    A['search_text'] = search_text
    
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
    
    from analysis.edgelist.kistep_edgelist import SnaEdgelist
    from analysis.word_extractors.topic_extractor import *
    from sna_functions import *
    from tree_functions import *

    import timeit
    import json
    from itertools import product

    start_time = timeit.default_timer()
    target_db = 'docu_test'
    search_text = '인공뉴런'
    
    filtering_dict = {'cdfreq': 1, 'dfreq': 1, 'node_limit_base': 1, 'root_parent_base': 400,
                  't_cdfreq': 1, 'term1_limit': 100, 'term_ratio': 0.1, 'weight_filter': 1}
    
    target_db_list = ['docu_test', 'docu_patent', 'docu_thesis', 'docu_subject']

    # # all_edgelist = we_edge.get_all_edgelist(target_db=target_db)
    
    
    """
    * DB 들어오는 가정
        - DB 아래 구조를 만들기 위해서는 각 DB에서 뽑은 ego_edgelist와 분석 결과 중 topic_word
        - topic_word로 단어들을 한정시킴 / 즉, 나중에 tree module을 작동시킬 때 ego_edgelist와 topic_word가 넘어와야 한다.
        - dict 구조로 넘어온다고 가정하자

    * Input : sna 결과에서 받아옴
        - ego_edgelist
        - topic_word

    """
    # topic_word를 sna에서 받아오는 구조가 없어서, sna.py의 함수를 그대로 사용하여 topic_word 생성
    # 따라서 현재 시행시간은 sna.py + alpha, 그러나 topic_word만 받아오면 생성에 많은 시간이 걸리지 않는다.
    
    # 단어 필터링 위한 로직 필요
    
    A = tree_sna_main(search_text, target_db_list, size=10)

    # with open("./tree_sample.json", "w") as f: 
    #     json.dump(A, f)

    print(A['nodes'][0],A['nodes'][1])
    print(A['links'][0])
    print(len(A['outer_links']))    
    print(A['search_text'])
    print(A)
    print(timeit.default_timer()-start_time)        