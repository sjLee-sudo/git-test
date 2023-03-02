#%%
# from edap.analysis.word_extractors.topic_extractor import extract_bridge_word
import pandas as pd
import numpy as np
import itertools
from functools import reduce
import igraph
from copy import copy
from scipy import sparse
import math
import json

if __name__ != '__main__':
    from .sna_functions import *
    from analysis.edgelist.kistep_edgelist import SnaEdgelist, PlayerEdgelist
"""
Input data는 Edgelist, EgoEdgelist, ResizeEdgelist로 구분

* EgoEdgelist : 검색된 문서에 대한 엣지리스트
* ResizeEdgelist : 검색된 문서에서 주제어만 남긴 엣지리스트

# EgoEdgelist columns
word        type : string
doc_id      type : string
dtfreq      type : int
position    type : list
section     type : string
dfreq       type : int
col_type    type : string

word    doc_id  dtfreq  position    section     dfreq   col_types
수소    100203  5       [1,3,4]     A           2       content
자동차  100203  3       [1,4,7]     A           4       title

# ResizeEdgelist는 EgoEdgelist와 동일

# json 구조
- paper와 word의 구조는 동일

* nodes
id : 노드의 고유번호 | int
t : 노드의 이름 | chr
_neighbor : 2mode에서 해당 노드와 이웃하는 노드(단어의 경우 해당 단어가 등장하는 문서) | list
pyear : 출현연도 | list
sE : 아이겐벡터 중심성 | float
weight : 문서의 weight | float

* edges
source : source | int
target : target | int
v : 엣지 가중치 | float
force : 강제 연결여부 | chr * word_network 에서만 존재

# example
{'nodes': [{
    'id': 0,
    't': '0.0',
    '_neighbor' : ['100023', '100242']
    'pyear' : ['2020','2019],
    'sE': 0.2658595716,
    'weight': 0.642
   },
   {
    'id': 1354,
    't': '희석',
    '_neighbor' : ['100023', '100242']
    'pyear' : ['2020','2019],
    'sE': 0.2658595716,
    'weight': 0.642
}],
 'links': [{'source': '0.0', 'target': 'Si', 'v': 3, 'force':''},
  {'source': '0.0', 'target': '규소', 'v': 3, 'force':''},
  {'source': '0.0', 'target': '난연제', 'v': 1, 'force':'T}]
}


# paper_cent
doc_id에 대한 중심성 지표 계산 결과
형태 : pandas dataframe

doc_id  E       D       B       S
110239  0.4     0.2     0.3     0.5

E : 아이겐벡터 중심성
D : degree
B : betweeness
S : bm25 score

모든 값은 표준화 한 값이며, D,B는 별도의 표준화 방법(중심성 지표 표준화 방법)에 따라 표준화 하였음


# player_network

* nodes
id : 노드의 고유번호 | int
name : 수행기관(section값이 None인 경우에는 주관기관) | str
pyear : 과제 수행 연도 | list(str)
linklen : 연결중심성 | int
cluster : graph cluster id | int
section : 주관기관

* links
source : source | int
target : target | int
weight : 기관별 연결 강도(0~1) | float

"""

filtering_dict = {'cdfreq': 1, 'dfreq': 1, 'node_limit_base': 1, 'root_parent_base': 400,
                  't_cdfreq': 1, 'term1_limit': 100, 'term_ratio': 0.1, 'weight_filter': 1}

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

# def word_network(ResizeEdgelist, pmi_mat, w_dict, min_P=0.0, N=300):
    
#     e,n,g = df2edge(ResizeEdgelist, 'word')
#     P = np.array([pmi_mat[w_dict[r],w_dict[c]] for r,c,w in tqdm(e.values)])
#     e_reduce = e.iloc[np.nonzero(P>min_P)[0]]
#     g = make_one_mode_graph(e_reduce)
#     E = np.array(g.eigenvector_centrality())
#     T = [g.vs['name'][i] for i in E.argsort()[::-1]][0:N]
    
#     e_reduce = e_reduce[e_reduce.source.isin(T)]
#     e_reduce = e_reduce[e_reduce.target.isin(T)]
    
#     g = make_one_mode_graph(e_reduce)    
    
#     return g,T

def word_network(ResizeEdgelist, pmi_mat, w_dict, min_P = 0.0, min_force = 0.5, N = 300):

    resize_data = copy.copy(ResizeEdgelist)
    w_c_dict = {w:t for w,t in zip(resize_data['word'], resize_data['col_type'])}
    e, n, g = df2edge(resize_data, 'word')
 

    # 1. edgelist에 존재하는 type 정보를 이용하여 합성어를 찾는다.
    e['sc'] = e.source.map(w_c_dict)
    e['tc'] = e.target.map(w_c_dict)

    e_n = list(set(e[e['sc'] == 'compound']['source'].unique().tolist() + e[e['tc'] == 'compound']['target'].unique().tolist()))

    # # 합성어에는 공백이 포함되어 있으므로, 이를 이용하여 합성어를 찾는다. - 제거
    # e_n = set(e.source.unique().tolist() + e.target.unique().tolist())
    # e_n = [k for k in e_n if " " in k]     
    
    # 2. 합성어가 존재할 때 합성어를 이루는 단어들 간 평균 pmi값을 계산한다.
    # 인공지능, 자율주행, 자동차라면 (인공지능,자동차),(자율주행,자동차),(인공지능,자율주행) // 3가지 경우의 pmi 평균이다.
    # 즉, 가능한 조합의 pmi 평균을 구하는 것이다.
    # r_edge는 합성어가 포함된 edgelist(source, target, source/target)

    def list_convert(list_2d, list_n, mode = 'split'):
        list_1d = list(itertools.chain(*list_2d))

        if mode == 'split':
            list_1d = iter([w_dict[i] for i in list_1d])

            return [list(itertools.islice(list_1d, i)) for i in list_n]

        elif mode == 'mean':
            list_1d_r = [l[0] for l in list_1d]
            list_1d_c = [l[1] for l in list_1d]
            list_1d_v = iter(pmi_mat[list_1d_r, list_1d_c].tolist()[0])

            return [np.array(list((itertools.islice(list_1d_v, i)))).mean() for i in list_n]

    if e_n:

        # 공백 이용 index 찾기
        # c_idx = [i for i,s,t in zip(e.index, e.source,e.target) if " " in s+t]
        # e_c = e.iloc[c_idx]

        c_idx = e[(e.sc == 'compound') | (e.tc == 'compound')].index.tolist()
        e_c = e.iloc[c_idx]        
        
        sc = [s.split(' ') for s in e_c.source]
        sn = [len(s) for s in sc]
        tc = [t.split(' ') for t in e_c.target]
        tn = [len(s) for s in tc]

        r_edge = pd.DataFrame({'sc': sc,
                               'tc': tc,
                               'sn': sn,
                               'tn': tn}, index = c_idx)

        sc = list_convert(sc, sn)
        tc = list_convert(tc, tn)

        c_g_idx = [sorted(list(set(s + t))) for s,t in zip(sc, tc)]

        _cg = [list(itertools.combinations(i, 2)) for i in c_g_idx]
        _cg_n = [len(c) for c in _cg]

        _tc = [list(itertools.combinations(i, 2)) for i in tc]
        _tc_n = [len(t) for t in _tc]

        _cg = list_convert(_cg, _cg_n, mode = 'mean')
        _tc = list_convert(_tc, _tc_n, mode = 'mean')

        r_edge['mpmi'] = _cg
        r_edge['mpmi_t'] = _tc
        r_edge['pmir'] = r_edge['mpmi'] / r_edge['mpmi_t']
        r_edge = r_edge[r_edge['pmir'] != 1] # 포함관계 제거
        r_edge = r_edge[r_edge['pmir'] > 0.5] # 합성어 edge filtering

        # sc = r_edge.apply(lambda x: [w_dict[i] for i in x.sc], axis=1).tolist()
        # tc = r_edge.apply(lambda x: sorted([w_dict[i] for i in x.tc]), axis=1).tolist()
        # c_g_idx = [sorted(list(set(s+t))) for s,t in zip(sc, tc)]

        # _cg = pd.Series([list(itertools.combinations(i, 2)) for i in c_g_idx])
        # _tc = pd.Series([list(itertools.combinations(i, 2)) for i in tc])
        
        # r_edge['mpmi'] = _cg.apply(lambda x:np.array([pmi_mat[i] for i in x]).mean()).tolist()
        # r_edge['mpmi_t'] = _tc.apply(lambda x:np.array([pmi_mat[i] for i in x]).mean()).tolist()
        # r_edge['pmir'] = r_edge.mpmi / r_edge.mpmi_t
        # r_edge = r_edge[r_edge.pmir != 1] # 포함관계 제거
        # r_edge = r_edge[r_edge.pmir > 0.5] # 합성어 edge filtering

    _row, _col, _val = sparse.find(pmi_mat)

    mat_df = pd.DataFrame({'source': _row, 
                           'target': _col, 
                           'weight': _val})

    w_dict_inv = {v:i for i, v in zip(w_dict.keys(), w_dict.values())}

    mat_df['source'] = mat_df['source'].map(w_dict_inv)
    mat_df['target'] = mat_df['target'].map(w_dict_inv)
    e = e.drop('weight', axis = 1)
    e = pd.merge(e, mat_df, on = ['source', 'target'], how = 'left')

    # 3. 합성어에 최대 weight인 1을 부여하여(기존 weight는 0~1사이 표준화 pmi 값) 필터링 되지 않도록 만든다.
    # r_c_idx는 제거할 합성어의 index(remove compound index)
    # 위에서 r_edge(remain_edge), 즉 합성어 중 일정 기준 이상의 edge만 남겼다. 기준을 충족하지 못하는 edge는 제거한다.

    if e_n:
        # 합성어에 weight 1 부여
        e.loc[c_idx, 'weight'] = 1

        # 합성어가 포함된 조합 필터링
        r_c_idx = set(c_idx).difference(r_edge.index)
        r_c_idx = e.index.isin(r_c_idx)

        e = e[~r_c_idx]
        e = e.dropna()

    e_reduce = e[e['weight'] > 0]
    g = make_one_mode_graph(e_reduce)
    # E = np.array(g.eigenvector_centrality())
    E = np.array(g.strength(weights = e_reduce['weight']))
    T = [g.vs['name'][i] for i in E.argsort()[::-1]][0:N] + e_n
    # T = [g.vs['name'][i] for i in E.argsort()[::-1]][0:40] + e_n

    #### Node size   
    n_E = pd.DataFrame({'name': g.vs['name'],
                        'sE': E})
    n = pd.merge(n, n_E, on = 'name')
    C = (10 - 3) / (n['sE'].max() - n['sE'].min())
    D = 3 - C * n['sE'].min()

    n['s'] = np.round(C * n['sE'] + D, 0)
    ####

    # 아이겐벡터 중심성 기준으로 node와 edge 축소
    e_reduce = e_reduce[e_reduce['source'].isin(T)]
    e_reduce = e_reduce[e_reduce['target'].isin(T)]
    n_reduce = n[n['name'].isin(T)]
            
    ## force attribute
    # pmi 기준으로 filtering 되면서 연결이 끊어지고, 네트워크가 분리된다.
    # 이를 방지하기 위해 source 기준으로 최소한 하나의 연결은 유지되도록 한다.
    # 다만 너무 낮은 값의 edge는 굳이 유지할 필요가 없으므로 일정 기준(weight > 0.1) 이상의 연결만 유지시킨다.
    e_add = e_reduce.rename(columns = {'source': 'target', 'target': 'source'})
    e_add = pd.concat([e_reduce, e_add])

    # force 기준 상향(최소 weight값 지정)
    _force = e_add.groupby('source').apply(lambda x: x[x['weight'] >= min_force].sort_values('weight', ascending = False).sort_index().head(1).index.tolist()).tolist()
    _force = list(set(itertools.chain(*_force)))
    e_reduce['force'] = ['T' if s in _force else "" for s in e_reduce.index]
    
    # g_test = make_one_mode_graph(e_reduce)
    # E_test = np.array(g_test.degree())
    # print([g.vs['name'][i] for i in E_test.argsort()[::-1]][0:10])

    #### 추후 삭제 가능 부분 // 시각화 팀에서 요구하는 구조에 맞게 데이터 이름 변경
    e_reduce = e_reduce.rename(columns = {'weight': 'v'})

    n_reduce = n_reduce.drop(columns = ['cluster', 'linklen'])    
    n_reduce = n_reduce.rename(columns = {'name': 't'})    
    ####    

    # id encoding
    _dict = {i:v for i, v in zip(n_reduce['id'], n_reduce['t'])}
    e_reduce['source'] = e_reduce['source'].map({v:i for i, v in zip(_dict.keys(), _dict.values())})
    e_reduce['target'] = e_reduce['target'].map({v:i for i, v in zip(_dict.keys(), _dict.values())})
    e_reduce = e_reduce.drop(['sc', 'tc'], axis = 1)

    _json = make_json(n_reduce, e_reduce)
    with open("./word_normal.json", "w") as f:
        json.dump(_json, f)
    return T, _json


def word_network_bm(ResizeEdgelist, pmi_mat, w_dict, min_P = 0.0, min_force =0.5, N = 300):

    resize_data = copy.copy(ResizeEdgelist)
    e, n, g = df2edge(resize_data, 'word',_map=False)
    
    _row, _col, _val = sparse.find(pmi_mat)

    mat_df = pd.DataFrame({'source': _row, 
                           'target': _col, 
                           'weight': _val})
    list1=[]
    list2=[]
    #print(n.loc[:,['pyear']])
    for i,v in enumerate(n.iterrows()):
        list1.append(['2020','2021'])
        list2.append(['UK'])
    n['pyear']=list1
    n['section']=list2
    w_dict_inv = {v:i for i, v in zip(w_dict.keys(), w_dict.values())}




    mat_df['source'] = mat_df['source'].map(w_dict_inv)
    mat_df['target'] = mat_df['target'].map(w_dict_inv)
    e = e.drop('weight', axis = 1)
    e = pd.merge(e, mat_df, on = ['source', 'target'], how = 'left')
    
    e_reduce = e[e['weight'] > 0]
    g = make_one_mode_graph(e_reduce)

   # E = np.array(g.eigenvector_centrality())
    E = np.array(g.strength(weights = e_reduce['weight']))
    T = [g.vs['name'][i] for i in E.argsort()[::-1]][0:N]

    #### Node size   
    n_E = pd.DataFrame({'name': g.vs['name'],
                        'sE': E})
    n = pd.merge(n, n_E, on = 'name')
    C = (10 - 3) / (n['sE'].max() - n['sE'].min())
    D = 3 - C * n['sE'].min()

    n['s'] = np.round(C * n['sE'] + D, 0)
    ####

    # 아이겐벡터 중심성 기준으로 node와 edge 축소
    e_reduce = e_reduce[e_reduce['source'].isin(T)]
    e_reduce = e_reduce[e_reduce['target'].isin(T)]
    n_reduce = n[n['name'].isin(T)]
            
    ## force attribute
    # pmi 기준으로 filtering 되면서 연결이 끊어지고, 네트워크가 분리된다.
    # 이를 방지하기 위해 source 기준으로 최소한 하나의 연결은 유지되도록 한다.
    # 다만 너무 낮은 값의 edge는 굳이 유지할 필요가 없으므로 일정 기준(weight > 0.1) 이상의 연결만 유지시킨다.
    e_add = e_reduce.rename(columns = {'source': 'target', 'target': 'source'})
    e_add = pd.concat([e_reduce, e_add])
 
    # force 기준 상향(최소 weight값 지정)
    e_add['weight'] = e_add['weight'] / e_add['weight'].max()

    _force = e_add.groupby('source').apply(lambda x: x[x['weight'] >= min_force].sort_values('weight', ascending = False).sort_index().head(1).index.tolist()).tolist()
    _force = list(set(itertools.chain(*_force)))
    e_reduce['force'] = ['T' if s in _force else "" for s in e_reduce.index]
    # g_test = make_one_mode_graph(e_reduce)
    # E_test = np.array(g_test.degree())
    # print([g.vs['name'][i] for i in E_test.argsort()[::-1]][0:10])

    #### 추후 삭제 가능 부분 // 시각화 팀에서 요구하는 구조에 맞게 데이터 이름 변경
    e_reduce = e_reduce.rename(columns = {'weight': 'v'})

    n_reduce = n_reduce.drop(columns = ['cluster', 'linklen'])    
    n_reduce = n_reduce.rename(columns = {'name': 't'})    
    ####    

    # id encoding
    _dict = {i:v for i, v in zip(n_reduce['id'], n_reduce['t'])}
    e_reduce['source'] = e_reduce['source'].map({v:i for i, v in zip(_dict.keys(), _dict.values())})
    e_reduce['target'] = e_reduce['target'].map({v:i for i, v in zip(_dict.keys(), _dict.values())})

    _json = make_json(n_reduce, e_reduce)

    return T, _json


"""
# topic_word
- list 형식
topic_word = [인공지능, 컴퓨터, ...]

# issue
사용자가 중요하다고 생각하여 결과에서 제거된 단어를 추가할 때,
네트워크는 PMI 등으로 걸러지기 때문에 사용자가 단어를 추가한다고 해서 네트워크 결과에 반영되지 않을 것이다.
따라서 PMI로 걸러지기 전, refine_ego_edgelist를 기준으로 그려진 원본 그래프 결과를 가져올 필요가 있다.

* e columns
source  target  value
정답    사용자      1
반응    사용자      2

e_reduce와 n_reduce에 '사용자'의 edge와 node정보가 없다면 기존의 e, n에서 해당 정보를 가져와서 e_reduce와 n_reduce에 추가한다.
'사용자' 정보를 가져올 때, e에는 존재하지만 n_reduce에는 존재하지 않는 node가 있다면 해당 정보도 n에서 n_reduce로 가져온다.
사용자가 추가하는 단어가 많지 않을 것으로 가정하고, 몇 개의 단어가 추가된다고 해서 그래프가 심하게 복잡해지지 않을 것으로 예상한다.
다만 지나치게 일반적인 단어를 추가하는 경우 그래프의 복잡도가 커질 수 있다. 이 경우 e에서 정보를 가져올 때 연결되는 단어의 중요도 순으로 sorting한다.
- 예를 들어 새로 추가되는 edge가 n개 이상인 경우, 단어 중요도 순으로 n개의 edge만 추가한다.
- n_reduce에 없는 n을 추가하지 않고, 해당 연결은 삭제하는 것도 고려할 수 있다.

단계상으로는 e_reduce와 n_reduce가 생성되는 extract_topic_word 다음에 새로운 task를 만들어서 특수한 상황에 적용하는 것으로 한다.

"""

def paper_network(ResizeEdgelist, EgoEdgelist, topic_word, s_dict, _revert = 1):

    # 1. data load한 뒤 word_network에서 생성된 topic_word만을 문서 연결의 대상으로 설정
    resize_data = copy.copy(ResizeEdgelist)
    resize_data = resize_data[resize_data['word'].isin(topic_word)]
    
    # 2. revert(문서 간 연결 복구)를 위해 ego_edgelist로 네트워크 구조 생성
    ego_data = copy.copy(EgoEdgelist)
    # N = ego_data.doc_id.unique().size
    # ego_data['bm25'] = bm25(N,ego_data.dtfreq, ego_data.dfreq).rename('bm25')
    # term_indice_df = ego_edgelist_node_controll(ego_data, g, filtering_dict)
    # ego_data = complete_egoedgelist(ego_data, term_indice_df, filtering_dict)
    e_raw, n_raw, g_raw = df2edge(ego_data, 'paper', p_mode = 'max')
    
    # 3. pruning을 위해 각 문서 내 단어들에 대해 bm25 score를 구하고 이를 정렬 기준으로 잡는다.
    # 문서 건수(N)이 설정된 기준(700)보다 많으면 최대 가중치로 pruning이 진행된다.
    # 그렇지 않으면 그래프의 밀도를 계산해서 밀도를 기준으로 pruning이 진행된다.    
    dfreq = resize_data.groupby(['word']).size().rename('dfreq')
    resize_d_edgelist = pd.merge(resize_data, dfreq, left_on = 'word', right_index = True)
    
    N = resize_d_edgelist['doc_id'].unique().size
    resize_d_edgelist['bm25'] = bm25(N, resize_d_edgelist['dtfreq'], resize_d_edgelist['dfreq']).rename('bm25')
    
    # term_indice_df: resize_d_edgelist에서 모든 word에 대해 node_limit를 계산해서 최대 2까지만
    if N >= 700:
        g = 1
        term_indice_df = ego_edgelist_node_controll(resize_d_edgelist, g, filtering_dict, mode = 'max')

    else:    
        g = df2edge(resize_data, 'paper')[2]
        term_indice_df = ego_edgelist_node_controll(resize_d_edgelist, g, filtering_dict)
    
    resize_d_edgelist = complete_egoedgelist(resize_d_edgelist, term_indice_df, filtering_dict)

    e, n, g = df2edge(resize_d_edgelist, 'paper', p_mode = 'max')

    # 4. 네트워크 내에 생성된 군집의 중심점을 구하고(2-mode에서 연결된 edge의 수 기준), 해당 노드를 포함하는 연결을 살린다.
    n_cluster = n.loc[:, ['name', 'cluster']].set_index('name')
    t_cluster = n_cluster['cluster'].value_counts().rename_axis('label').reset_index(name='counts')
    label = t_cluster.loc[0, 'label']

    center_list = [n[n['cluster'] == c].sort_values('linklen', ascending = False).name.iloc[0:_revert].tolist() for c in range(1, max(n['cluster']))]
    center_list = list(itertools.chain(*center_list))    # cluster의 중심점 list

    e_revert = pd.merge(e_raw, n_cluster, left_on = 'source', right_index = True)
    e_revert = pd.merge(e_revert, n_cluster, left_on = 'target', right_index = True)
    e_revert = e_revert[(e_revert['source'].isin(center_list)) | (e_revert['target'].isin(center_list))]
    
    # drop_duplicate를 하기 전, shuffling을 통해서 일정 Node만 가져오는 현상 해결
    e_revert = e_revert[(e_revert['cluster_x'] == label) | (e_revert['cluster_y'] == label)].sample(frac = 1, random_state = 42).drop_duplicates(['cluster_x', 'cluster_y'])
    # e_revert_source = e_revert[e_revert['cluster_x'] == label].drop_duplicates(['cluster_x', 'cluster_y'], keep = 'last')
    # e_revert_target = e_revert[e_revert['cluster_y'] == label].drop_duplicates(['cluster_x', 'cluster_y'], keep = 'first')
    # e_revert = pd.concat([e_revert_source, e_revert_target], axis = 0)
    e_revert = e_revert.drop(['cluster_x', 'cluster_y'], axis = 1)

    e = pd.concat([e, e_revert]).reset_index(drop = True)
    e['force'] = ''

    # 5. 각 노드들의 중심성지표 값을 포함하는 dataframe을 생성한다.
    # N=2 인 경우의 문제로 일단 표준화 중지

    g = make_one_mode_graph(e)

    # Centrality score
    E = np.array(g.eigenvector_centrality())
    # E = np.array(g.strength(weights = e['weight']))
    D = np.array(g.degree())
    # D = np.array(g.degree()) / (N-1)
    B = g.betweenness()
    # B = [b*2/((N-1)*(N-2)) for b in B]
    
    g_cent = pd.DataFrame({'doc_id': g.vs['name'], 
                           'E': E, 
                           'D': D, 
                           'B': B})    

    # n = pd.merge(n,E,left_on = 'name', right_index = True)

    e = e.rename(columns = {'weight': 'v'})   # v
    # e['v'] = (e['v'] - e['v'].min()) / (e['v'].max() - e['v'].min())
    e['v'] = e['v'] / e['v'].max()

    n = n.drop(columns = ['cluster', 'linklen'])
    n = n.rename(columns = {'name': 'doc_id'})

    n = pd.merge(n, resize_data.loc[:, ['doc_id', 'title']].drop_duplicates(), on = 'doc_id', how = 'left')
    n = n.rename(columns = {'title': 't'})
    
    # node weight 
    _score = pd.DataFrame({'doc_id': s_dict.keys(),
                           'weight': s_dict.values()})
    _score['weight'] = _score['weight'] / _score['weight'].max()

    #### Node size    
    n_E = g_cent[['doc_id', 'E']]
    n = pd.merge(n, n_E, on = 'doc_id')
    C = (20 - 3) / (n['E'].max() - n['E'].min())
    D = 3 - C * n['E'].min()

    n['s'] = np.round(C * n['E'] + D, 0)
    ####

    n = pd.merge(n, _score, on = 'doc_id')

    n = n.drop(['E'], axis = 1)
    # n = n.rename(columns = {'weight': 'b'})   # s

    g_cent = pd.merge(g_cent, _score, on = 'doc_id')
    g_cent = g_cent.rename(columns = {'weight': 'S'})
    g_cent = g_cent.fillna(0)
    # T = [g.vs['name'][i] for i in E.argsort()[::-1]][0:300]
    
    # # 아이겐벡터 중심성 기준으로 node와 edge 축소
    # e = e[e.source.isin(T)]
    # e = e[e.target.isin(T)]    
    # n = n[n.doc_id.isin(T)]

    # id encoding

    _dict = {i:v for i, v in zip(n['id'], n['doc_id'])}    

    e['source'] = e['source'].map({v:i for i,v in zip(_dict.keys(), _dict.values())})
    e['target'] = e['target'].map({v:i for i,v in zip(_dict.keys(), _dict.values())})

    _json = make_json(n, e)
    
    return _json, g_cent


def paper_network_bm(ResizeEdgelist, EgoEdgelist, topic_word, s_dict, _revert = 1):
    # 1. data load한 뒤 word_network에서 생성된 topic_word만을 문서 연결의 대상으로 설정

    resize_data = copy.copy(ResizeEdgelist)
    resize_data = resize_data[resize_data['word'].isin(topic_word)]

    # 2. revert(문서 간 연결 복구)를 위해 ego_edgelist로 네트워크 구조 생성
    ego_data = copy.copy(EgoEdgelist)


    # N = ego_data.doc_id.unique().size
    # ego_data['bm25'] = bm25(N,ego_data.dtfreq, ego_data.dfreq).rename('bm25')
    # term_indice_df = ego_edgelist_node_controll(ego_data, g, filtering_dict)
    # ego_data = complete_egoedgelist(ego_data, term_indice_df, filtering_dict)

    e_raw, n_raw, g_raw = df2edge(ego_data, 'paper',_map = False, p_mode = 'max')
    # 3. pruning을 위해 각 문서 내 단어들에 대해 bm25 score를 구하고 이를 정렬 기준으로 잡는다.
    # 문서 건수(N)이 설정된 기준(700)보다 많으면 최대 가중치로 pruning이 진행된다.
    # 그렇지 않으면 그래프의 밀도를 계산해서 밀도를 기준으로 pruning이 진행된다.    
    dfreq = resize_data.groupby(['word']).size().rename('dfreq')
    resize_d_edgelist = pd.merge(resize_data, dfreq, left_on = 'word', right_index = True)
    
    N = resize_d_edgelist['doc_id'].unique().size
    resize_d_edgelist['bm25'] = bm25(N, resize_d_edgelist['dtfreq'], resize_d_edgelist['dfreq']).rename('bm25')
    
    # term_indice_df: resize_d_edgelist에서 모든 word에 대해 node_limit를 계산해서 최대 2까지만
    if N >= 700:
        g = 1
        term_indice_df = ego_edgelist_node_controll(resize_d_edgelist, g, filtering_dict, mode = 'max')

    else:    
        g = df2edge(resize_data, 'paper',_map = False)[2]
        term_indice_df = ego_edgelist_node_controll(resize_d_edgelist, g, filtering_dict)
    
    resize_d_edgelist = complete_egoedgelist(resize_d_edgelist, term_indice_df, filtering_dict)
    
    e, n, g = df2edge(resize_d_edgelist, 'paper', _map = False, p_mode = 'max')
 
    # 4. 네트워크 내에 생성된 군집의 중심점을 구하고(2-mode에서 연결된 edge의 수 기준), 해당 노드를 포함하는 연결을 살린다.
    n_cluster = n.loc[:, ['name', 'cluster']].set_index('name')
    t_cluster = n_cluster['cluster'].value_counts().rename_axis('label').reset_index(name='counts')
    label = t_cluster.loc[0, 'label']

    center_list = [n[n['cluster'] == c].sort_values('linklen', ascending = False).name.iloc[0:_revert].tolist() for c in range(1, max(n['cluster']))]
    center_list = list(itertools.chain(*center_list))    # cluster의 중심점 list

    e_revert = pd.merge(e_raw, n_cluster, left_on = 'source', right_index = True)
    e_revert = pd.merge(e_revert, n_cluster, left_on = 'target', right_index = True)
    e_revert = e_revert[(e_revert['source'].isin(center_list)) | (e_revert['target'].isin(center_list))]

    # drop_duplicate를 하기 전, shuffling을 통해서 일정 Node만 가져오는 현상 해결
    e_revert = e_revert[(e_revert['cluster_x'] == label) | (e_revert['cluster_y'] == label)].sample(frac = 1, random_state = 42).drop_duplicates(['cluster_x', 'cluster_y'])
    # e_revert_source = e_revert[e_revert['cluster_x'] == label].drop_duplicates(['cluster_x', 'cluster_y'], keep = 'last')
    # e_revert_target = e_revert[e_revert['cluster_y'] == label].drop_duplicates(['cluster_x', 'cluster_y'], keep = 'first')
    # e_revert = pd.concat([e_revert_source, e_revert_target], axis = 0)
    e_revert = e_revert.drop(['cluster_x', 'cluster_y'], axis = 1)

    e = pd.concat([e, e_revert]).reset_index(drop = True)


    e['force'] = ''

    # 5. 각 노드들의 중심성지표 값을 포함하는 dataframe을 생성한다.
    # N=2 인 경우의 문제로 일단 표준화 중지

    g = make_one_mode_graph(e)

    # Centrality score
    E = np.array(g.eigenvector_centrality())
    # E = np.array(g.strength(weights = e['weight']))
    D = np.array(g.degree())
    # D = np.array(g.degree()) / (N-1)
    B = g.betweenness()
    # B = [b*2/((N-1)*(N-2)) for b in B]
    
    g_cent = pd.DataFrame({'doc_id': g.vs['name'], 
                           'E': E, 
                           'D': D, 
                           'B': B})    

    # n = pd.merge(n,E,left_on = 'name', right_index = True)

    e = e.rename(columns = {'weight': 'v'})   # v
    # e['v'] = (e['v'] - e['v'].min()) / (e['v'].max() - e['v'].min())
    e['v'] = e['v'] / e['v'].max()

    n = n.drop(columns = ['cluster', 'linklen'])
    n = n.rename(columns = {'name': 'doc_id'})
    
    n = pd.merge(n, resize_data.loc[:, ['doc_id', 'title']].drop_duplicates(), on = 'doc_id', how = 'left')
    n = n.rename(columns = {'title': 't'})

    # node weight 
    _score = pd.DataFrame({'doc_id': s_dict.keys(),
                           'weight': s_dict.values()})


    #### Node size    
    n_E = g_cent[['doc_id', 'E']]
    n = pd.merge(n, n_E, on = 'doc_id')
    C = (20 - 3) / (n['E'].max() - n['E'].min())
    D = 3 - C * n['E'].min()

    n['s'] = np.round(C * n['E'] + D, 0)
    n = pd.merge(n, _score, on = 'doc_id')

    n = n.drop(['E'], axis = 1)
    # n = n.rename(columns = {'weight': 'b'})   # s
   
    g_cent = pd.merge(g_cent, _score, on = 'doc_id')
    g_cent = g_cent.rename(columns = {'weight': 'S'})
    g_cent = g_cent.fillna(0)
    # T = [g.vs['name'][i] for i in E.argsort()[::-1]][0:300]
    
    # # 아이겐벡터 중심성 기준으로 node와 edge 축소
    # e = e[e.source.isin(T)]
    # e = e[e.target.isin(T)]    
    # n = n[n.doc_id.isin(T)]

    # id encoding

    _dict = {i:v for i, v in zip(n['id'], n['doc_id'])}    

    e['source'] = e['source'].map({v:i for i,v in zip(_dict.keys(), _dict.values())})
    e['target'] = e['target'].map({v:i for i,v in zip(_dict.keys(), _dict.values())})
    list1=[]
    list2=[]

    for i,v in enumerate(n.iterrows()):
        list1.append(['2020','2021'])
        list2.append(['UK'])
    n['pyear']=list1
    n['section']=list2
    _json = make_json(n, e)
    

    return _json, g_cent

def paper_network_bm_new(ResizeEdgelist, EgoEdgelist, topic_word, s_dict, _revert = 1):
    # 1. data load한 뒤 word_network에서 생성된 topic_word만을 문서 연결의 대상으로 설정

    resize_data = copy.copy(ResizeEdgelist)
    resize_data = resize_data[resize_data['word'].isin(topic_word)]

    # 2. revert(문서 간 연결 복구)를 위해 ego_edgelist로 네트워크 구조 생성
    ego_data = copy.copy(EgoEdgelist)


    # N = ego_data.doc_id.unique().size
    # ego_data['bm25'] = bm25(N,ego_data.dtfreq, ego_data.dfreq).rename('bm25')
    # term_indice_df = ego_edgelist_node_controll(ego_data, g, filtering_dict)
    # ego_data = complete_egoedgelist(ego_data, term_indice_df, filtering_dict)

    e_raw, n_raw, g_raw = df2edge(ego_data, 'paper',_map = False, p_mode = 'max')
    # 3. pruning을 위해 각 문서 내 단어들에 대해 bm25 score를 구하고 이를 정렬 기준으로 잡는다.
    # 문서 건수(N)이 설정된 기준(700)보다 많으면 최대 가중치로 pruning이 진행된다.
    # 그렇지 않으면 그래프의 밀도를 계산해서 밀도를 기준으로 pruning이 진행된다.   

    dfreq = resize_data.groupby(['word']).size().rename('dfreq')
    resize_d_edgelist = pd.merge(resize_data, dfreq, left_on = 'word', right_index = True)
    
    N = resize_d_edgelist['doc_id'].unique().size

                   
    # term_indice_df: resize_d_edgelist에서 모든 word에 대해 node_limit를 계산해서 최대 2까지만
    if N >= 700:
        g = 1
        term_indice_df = ego_edgelist_node_controll(resize_d_edgelist, g, filtering_dict, mode = 'max')
        resize_d_edgelist = complete_egoedgelist(resize_d_edgelist, term_indice_df, filtering_dict)
    elif N>250 :
        
        g = df2edge(resize_data, 'paper',_map = False)[2]
        term_indice_df = ego_edgelist_node_controll(resize_d_edgelist, g, filtering_dict)
        resize_d_edgelist = complete_egoedgelist(resize_d_edgelist, term_indice_df, filtering_dict)
    e, n, g = df2edge(resize_d_edgelist, 'paper', _map = False, p_mode = 'max')
 
    # 4. 네트워크 내에 생성된 군집의 중심점을 구하고(2-mode에서 연결된 edge의 수 기준), 해당 노드를 포함하는 연결을 살린다.
    n_cluster = n.loc[:, ['name', 'cluster']].set_index('name')
    t_cluster = n_cluster['cluster'].value_counts().rename_axis('label').reset_index(name='counts')
    label = t_cluster.loc[0, 'label']

    center_list = [n[n['cluster'] == c].sort_values('linklen', ascending = False).name.iloc[0:_revert].tolist() for c in range(1, max(n['cluster']))]
    center_list = list(itertools.chain(*center_list))    # cluster의 중심점 list

    e_revert = pd.merge(e_raw, n_cluster, left_on = 'source', right_index = True)
    e_revert = pd.merge(e_revert, n_cluster, left_on = 'target', right_index = True)
    e_revert = e_revert[(e_revert['source'].isin(center_list)) | (e_revert['target'].isin(center_list))]

    # drop_duplicate를 하기 전, shuffling을 통해서 일정 Node만 가져오는 현상 해결
    e_revert = e_revert[(e_revert['cluster_x'] == label) | (e_revert['cluster_y'] == label)].sample(frac = 1, random_state = 42).drop_duplicates(['cluster_x', 'cluster_y'])
    # e_revert_source = e_revert[e_revert['cluster_x'] == label].drop_duplicates(['cluster_x', 'cluster_y'], keep = 'last')
    # e_revert_target = e_revert[e_revert['cluster_y'] == label].drop_duplicates(['cluster_x', 'cluster_y'], keep = 'first')
    # e_revert = pd.concat([e_revert_source, e_revert_target], axis = 0)
    e_revert = e_revert.drop(['cluster_x', 'cluster_y'], axis = 1)

    e = pd.concat([e, e_revert]).reset_index(drop = True)


    e['force'] = ''

    # 5. 각 노드들의 중심성지표 값을 포함하는 dataframe을 생성한다.
    # N=2 인 경우의 문제로 일단 표준화 중지

    g = make_one_mode_graph(e)

    # Centrality score
    E = np.array(g.eigenvector_centrality())
    # E = np.array(g.strength(weights = e['weight']))
    D = np.array(g.degree())
    # D = np.array(g.degree()) / (N-1)
    B = g.betweenness()
    # B = [b*2/((N-1)*(N-2)) for b in B]
    
    g_cent = pd.DataFrame({'doc_id': g.vs['name'], 
                           'E': E, 
                           'D': D, 
                           'B': B})    

    # n = pd.merge(n,E,left_on = 'name', right_index = True)

    e = e.rename(columns = {'weight': 'v'})   # v
    # e['v'] = (e['v'] - e['v'].min()) / (e['v'].max() - e['v'].min())
    e['v'] = e['v'] / e['v'].max()

    n = n.drop(columns = ['cluster', 'linklen'])
    n = n.rename(columns = {'name': 'doc_id'})
    
    n = pd.merge(n, resize_data.loc[:, ['doc_id', 'title']].drop_duplicates(), on = 'doc_id', how = 'left')
    n = n.rename(columns = {'title': 't'})

    # node weight 
    _score = pd.DataFrame({'doc_id': s_dict.keys(),
                           'weight': s_dict.values()})


    #### Node size    
    n_E = g_cent[['doc_id', 'E']]
    n = pd.merge(n, n_E, on = 'doc_id')
    C = (20 - 3) / (n['E'].max() - n['E'].min())
    D = 3 - C * n['E'].min()

    n['s'] = np.round(C * n['E'] + D, 0)
    n = pd.merge(n, _score, on = 'doc_id')

    n = n.drop(['E'], axis = 1)
    # n = n.rename(columns = {'weight': 'b'})   # s
   
    g_cent = pd.merge(g_cent, _score, on = 'doc_id')
    g_cent = g_cent.rename(columns = {'weight': 'S'})
    g_cent = g_cent.fillna(0)
    # T = [g.vs['name'][i] for i in E.argsort()[::-1]][0:300]
    
    # # 아이겐벡터 중심성 기준으로 node와 edge 축소
    # e = e[e.source.isin(T)]
    # e = e[e.target.isin(T)]    
    # n = n[n.doc_id.isin(T)]

    # id encoding

    _dict = {i:v for i, v in zip(n['id'], n['doc_id'])}    

    e['source'] = e['source'].map({v:i for i,v in zip(_dict.keys(), _dict.values())})
    e['target'] = e['target'].map({v:i for i,v in zip(_dict.keys(), _dict.values())})
    list1=[]
    list2=[]

    for i,v in enumerate(n.iterrows()):
        list1.append(['2020','2021'])
        list2.append(['UK'])
    n['pyear']=list1
    n['section']=list2
    _json = make_json(n, e)
    

    return _json, g_cent


def player_network_original(PlayerEdgelist, _tree = True):

    _mgnt = PlayerEdgelist.loc[:, ['mgnt_org_cd', 'mgnt_org_nm']].rename(columns = {'mgnt_org_cd': 'code', 'mgnt_org_nm': 'name'}).drop_duplicates()
    _prfrm = PlayerEdgelist.loc[:, ['prfrm_org_cd', 'prfrm_org_nm']].rename(columns = {'prfrm_org_cd': 'code', 'prfrm_org_nm': 'name'}).drop_duplicates()
    _code = pd.concat([_mgnt, _prfrm]).drop_duplicates()
    real_name_dict = {c:n for n, c in zip(_code['name'], _code['code'])}

    result = PlayerEdgelist.rename(columns = {'mgnt_org_cd': 'section', 'prfrm_org_cd': 'orgn_id', 'term': 'word'})

    if not _tree:
        result_net = result.loc[:, ['orgn_id', 'section']]
        result_net = result_net.drop_duplicates().reset_index(drop = True)
        result_net = result_net.rename(columns = {'orgn_id': 'word', 'section': 'doc_id'})

        e,n,g = df2edge(result_net, 'word', _map = False)

        return e, n, g
    
    ### 이 부분을 나중에 연결 들어오면 수정하는 것으로 / 과제-수행기관 2-mode로 전환
    ### word > 수행기관으로 바꾼다고 생각하면 d_ratio를 삭제하면 문제 없음
    e, n, g = df2edge(result, 'paper', p_mode = 'max')
    # e_sec, n_sec, g_sec = df2edge_player(result, p_mode = 'max')

    d_size = result.groupby('doc_id').size().rename('d_size')

    e = pd.merge(e, d_size, left_on = 'source', right_index = True)

    # d_size는 각 문서에 포함된 단어의 수
    # e의 weight는 두 문서가 공유하는 단어의 수
    # 두 값의 비율을 새로운 weight(기관별 연결 강도)로 두고 filtering 기준으로 사용
    e['weight'] = e['weight'] / e['d_size']
    e['source'] = e['source'].map({v:i for i, v in zip(result['orgn_id'], result['doc_id'])})
    e['target'] = e['target'].map({v:i for i, v in zip(result['orgn_id'], result['doc_id'])})

    # e_sec['weight'] = (e_sec['weight'] - 0.001) / e_sec['weight'].max()
    
    e = e.drop_duplicates().loc[:, ['source', 'target', 'weight']]

    # e['d_ratio'] = e.weight / e.d_size    
    # e_new = e[e.d_ratio > u_limit] 
    # # filtering 기준으로도 사용 가능 // u_limit를 그냥 value로 넣어주는 식으로
    # # 나중에 바꿀때는 e_new 파트만 지워주고 d_ratio를 weight로 채우면 됨
    # e_new = e_new.copy() # ignore warning    
    # e_new.source = e_new.source.map({v:i for i,v in zip(result.orgn_id, result.doc_id)})
    # e_new.target = e_new.target.map({v:i for i,v in zip(result.orgn_id, result.doc_id)})
    # e = e_new.drop_duplicates().loc[:,['source','target','weight']]

    n['name'] = n['name'].map({v:i for i, v in zip(result['orgn_id'], result['doc_id'])})
    n = n.drop_duplicates('name')
    n = n.reset_index(drop = True)
    n = n.drop('id', axis = 1)
    n['section'] = [s[0] for s in n['section']]
    
    # doc_id 리스트 추가
    _doc_id_by_orgn_df = result.drop_duplicates(['orgn_id','doc_id']).groupby(['orgn_id'])['doc_id'].agg(list).reset_index().rename({'orgn_id':'name'},axis=1)
    n = n.merge(_doc_id_by_orgn_df,how='left')

    n_add = pd.DataFrame({'name': sorted(n['section'].unique())})    # 주관기관
    # n_add = pd.DataFrame({'name': sorted(n['name'].unique())})    # 수행기관
    
    _pyear = n.groupby('section').apply(lambda x: list(sorted(set(list(itertools.chain(*[a for a in x['pyear']])))))).rename('pyear')
    _linklen = n.groupby('section').size().rename('linklen')
    _doc_id_by_mgnt_orgn_df = result.drop_duplicates(['section','doc_id']).groupby(['section'])['doc_id'].agg(list)
    n_add = reduce(lambda left,right: pd.merge(left, right, left_on = 'name', right_index = True), [n_add, _pyear, _linklen, _doc_id_by_mgnt_orgn_df])
    n_add['cluster'] = 0
        
    e_add = n.loc[:, ['section', 'name']].rename(columns = {'section': 'source', 'name': 'target'})
    e_add['weight'] = 1

    # e = pd.concat([e, e_add]).reset_index(drop = True)
    
    e = e_add    # section, orgn link만 가져오도록
    e = e[e['source'] != e['target']]
    
    # remove loop
    e = e.drop_duplicates(['source', 'target'])
    e_dup = e.rename(columns = {'source': 'target', 'target': 'source'}).loc[:, ['source', 'target', 'weight']]
    e = e[~e.apply(tuple, 1).isin(e_dup.apply(tuple, 1))]

    n = pd.concat([n_add, n])
    n = n.reset_index(drop = True)
    n = n.reset_index().rename(columns = {'index': 'id'})

    # id encoding
    _dict = {i:v for i, v in zip(n['id'], n['name'])}
    e['source'] = e['source'].map({v:k for k, v in zip(_dict.keys(), _dict.values())})
    e['target'] = e['target'].map({v:k for k, v in zip(_dict.keys(), _dict.values())})

    n['name'] = n['name'].map(real_name_dict)
    # e.to_csv('player_net_edge.csv', encoding='utf-8-sig')
    # n.to_csv('player_net_node.csv', encoding='utf-8-sig')

    #### 
    n_section = n[n['section'].isnull()].copy()
    C = (20 - 6) / (n_section['linklen'].max() - n_section['linklen'].min())
    D = 6 - C * n_section['linklen'].min()

    n_section['s'] = np.round(C * n_section['linklen'] + D, 0)

    n_orgn = n[~n['section'].isnull()].copy()
    n_orgn['s'] = 3

    n_node_size = pd.concat([n_section['s'], n_orgn['s']], axis = 0)
    n = pd.concat([n, n_node_size], axis = 1)

    e = e.rename(columns = {'weight': 'v'})
    n = n.rename(columns = {'name': 't'})
    ####
    
    _json = make_json(n, e)   

    return _json
def doc_word_bm(PlayerEdgelist, _tree = True):
    

    PlayerEdgelist['pyear']='2020'
    _mgnt = PlayerEdgelist.loc[:, ['mgnt_org_cd', 'mgnt_org_nm']].rename(columns = {'mgnt_org_cd': 'code', 'mgnt_org_nm': 'name'}).drop_duplicates()
    _prfrm = PlayerEdgelist.loc[:, ['prfrm_org_cd', 'prfrm_org_nm']].rename(columns = {'prfrm_org_cd': 'code', 'prfrm_org_nm': 'name'}).drop_duplicates()
    _code = pd.concat([_mgnt, _prfrm]).drop_duplicates()
    real_name_dict = {c:n for n, c in zip(_code['name'], _code['code'])}

    result = PlayerEdgelist.rename(columns = {'mgnt_org_cd': 'section', 
                                              'prfrm_org_cd': 'orgn_id', 
                                              'term': 'word'})

    if not _tree:
        result_net = result.loc[:, ['orgn_id', 'section']]
        result_net = result_net.drop_duplicates().reset_index(drop = True)
        result_net = result_net.rename(columns = {'orgn_id': 'word', 
                                                  'section': 'doc_id'})

        e, n, g = df2edge(result_net, 'word', _map = False)

        return e, n, g

    e_orgn, n_orgn, _ = df2edge_player(result, 'orgn', drop_match = False, p_mode = 'max')
    e_mgnt, n_mgnt, _ = df2edge_player(result, 'mgnt', drop_match = False, p_mode = 'max')
    n_mgnt['type']=1
    n_orgn['type']=2
    

    # 수행기관과 주관기관이 둘 다 주관기관이고 같은 경우 - drop
    # 수행기관과 주관기관이 둘 다 주관기관이고 다른 경우 - keep
    e2 = pd.DataFrame({'source': result['orgn_id'],
                       'target': result['section']})
    e2 = e2.drop_duplicates(['source', 'target']).reset_index(drop = True)    

    weight = result.groupby(['section', 'orgn_id'])['doc_id'].nunique().agg(list)
    weight = pd.DataFrame(weight, columns = ['weight'])
    e2['weight'] = weight

    e = pd.concat([e2, e_mgnt], axis = 0)
    e = e[e['source'] != e['target']]
    e['weight'] = e['weight'] / e['weight'].max()
    n_orgn = n_orgn.drop('id', axis = 1)

    # 수행기관 doc_id list 추가
    _doc_id_by_orgn_df = result.drop_duplicates(['orgn_id', 'doc_id'])
    _doc_id_by_orgn_df = _doc_id_by_orgn_df.groupby(['orgn_id'])['doc_id'].agg(list).reset_index().rename({'orgn_id': 'name'}, axis = 1)
    n_orgn = n_orgn.merge(_doc_id_by_orgn_df, how = 'left')

    # 주관기관 doc_id list 추가
    _doc_id_by_mgnt_orgn_df = result.drop_duplicates(['section', 'doc_id'])
    _doc_id_by_mgnt_orgn_df = _doc_id_by_mgnt_orgn_df.groupby(['section'])['doc_id'].agg(list).reset_index().rename({'section': 'name'}, axis = 1)
    n_mgnt = n_mgnt.merge(_doc_id_by_mgnt_orgn_df, how = 'left')
    n_mgnt = n_mgnt.drop('id', axis = 1)

    n = pd.concat([n_mgnt, n_orgn], axis = 0).reset_index(drop = True)
    n = n.drop_duplicates(['name']).reset_index(drop = True)    # 수행기관 역할을 하는 주관기관 drop (주관기관 node가 위에 있으므로 keep = first)
    n = n.reset_index().rename(columns = {'index': 'id'})

    # id encoding
    _dict = {i:v for i, v in zip(n['id'], n['name'])}
    e['source'] = e['source'].map({v:k for k, v in zip(_dict.keys(), _dict.values())})
    e['target'] = e['target'].map({v:k for k, v in zip(_dict.keys(), _dict.values())})
    
    n['name'] = n['name'].map(real_name_dict)


    n_mgnt = n[n['section'].isnull()].copy()
    C = (10 - 6) / (n_mgnt['linklen'].max() - n_mgnt['linklen'].min())
    D = 6 - C * n_mgnt['linklen'].min()
    n_mgnt['s'] = np.round(C * n_mgnt['linklen'] + D, 0)

    n_orgn = n[~n['section'].isnull()].copy()
    n_orgn['s'] = n_orgn['doc_id'].apply(lambda x: len(x))
    if (n_orgn['s'].max() - n_orgn['s'].min())!=0 :
        C = (10 - 3) / (n_orgn['s'].max() - n_orgn['s'].min())
    else :
        C = 10
    D = 3 - C * n_orgn['s'].min()
    n_orgn['s'] = np.round(C * n_orgn['s'] + D, 0)

    n_node_size = pd.concat([n_mgnt['s'], n_orgn['s']], axis = 0)
    n_node_size.columns = ['s']
    n = pd.concat([n, n_node_size], axis = 1)

    ## rename
    e = e.rename(columns = {'weight': 'v'})
    n = n.rename(columns = {'name': 't'})
    
    _json = make_json(n, e)
    
    return _json


def orgn_network(PlayerEdgelist, _tree = True):
    _mgnt = PlayerEdgelist.loc[:, ['mgnt_org_cd', 'mgnt_org_nm']].rename(columns = {'mgnt_org_cd': 'code', 'mgnt_org_nm': 'name'}).drop_duplicates()
    _prfrm = PlayerEdgelist.loc[:, ['prfrm_org_cd', 'prfrm_org_nm']].rename(columns = {'prfrm_org_cd': 'code', 'prfrm_org_nm': 'name'}).drop_duplicates()
    _code = pd.concat([_mgnt, _prfrm]).drop_duplicates()
    real_name_dict = {c:n for n, c in zip(_code['name'], _code['code'])}
    result = PlayerEdgelist.rename(columns = {'mgnt_org_cd': 'section', 
                                              'prfrm_org_cd': 'orgn_id', 
                                              'term': 'word'})
    if not _tree:
        result_net = result.loc[:, ['orgn_id', 'section']]
        result_net = result_net.drop_duplicates().reset_index(drop = True)
        result_net = result_net.rename(columns = {'orgn_id': 'word', 
                                                  'section': 'doc_id'})

        e, n, g = df2edge(result_net, 'word', _map = False)

        return e, n, g

    e_orgn, n_orgn, _ = df2edge_player(result, 'orgn', drop_match = True, p_mode = 'max')
    e_mgnt, n_mgnt, _ = df2edge_player(result, 'mgnt', drop_match = True, p_mode = 'max')
    n_mgnt['type']=1
    n_orgn['type']=2

    # 수행기관과 주관기관이 둘 다 주관기관이고 같은 경우 - drop
    # 수행기관과 주관기관이 둘 다 주관기관이고 다른 경우 - keep
    e2 = pd.DataFrame({'source': result['orgn_id'],
                       'target': result['section']})
    e2 = e2.drop_duplicates(['source', 'target']).reset_index(drop = True)    

    weight = result.groupby(['section', 'orgn_id'])['doc_id'].nunique().agg(list)
    weight = pd.DataFrame(weight, columns = ['weight'])
    e2['weight'] = weight

    e = pd.concat([e2, e_mgnt], axis = 0)
    e = e[e['source'] != e['target']]
    e['weight'] = e['weight'] / e['weight'].max()

    n_orgn['section'] = [s[0] for s in n_orgn['section']]
    n_orgn = n_orgn.drop('id', axis = 1)

    # 수행기관 doc_id list 추가
    _doc_id_by_orgn_df = result.drop_duplicates(['orgn_id', 'doc_id'])
    _doc_id_by_orgn_df = _doc_id_by_orgn_df.groupby(['orgn_id'])['doc_id'].agg(list).reset_index().rename({'orgn_id': 'name'}, axis = 1)
    n_orgn = n_orgn.merge(_doc_id_by_orgn_df, how = 'left')

    # 주관기관 doc_id list 추가
    _doc_id_by_mgnt_orgn_df = result.drop_duplicates(['section', 'doc_id'])
    _doc_id_by_mgnt_orgn_df = _doc_id_by_mgnt_orgn_df.groupby(['section'])['doc_id'].agg(list).reset_index().rename({'section': 'name'}, axis = 1)
    n_mgnt = n_mgnt.merge(_doc_id_by_mgnt_orgn_df, how = 'left')
    n_mgnt = n_mgnt.drop('id', axis = 1)

    n = pd.concat([n_mgnt, n_orgn], axis = 0).reset_index(drop = True)
    n = n.drop_duplicates(['name']).reset_index(drop = True)    # 수행기관 역할을 하는 주관기관 drop (주관기관 node가 위에 있으므로 keep = first)
    n = n.reset_index().rename(columns = {'index': 'id'})

    # id encoding
    _dict = {i:v for i, v in zip(n['id'], n['name'])}
    e['source'] = e['source'].map({v:k for k, v in zip(_dict.keys(), _dict.values())})
    e['target'] = e['target'].map({v:k for k, v in zip(_dict.keys(), _dict.values())})
    
    n['name'] = n['name'].map(real_name_dict)

    # Node size
    n_mgnt = n[n['section'].isnull()].copy()
    C = (10 - 6) / (n_mgnt['linklen'].max() - n_mgnt['linklen'].min())
    D = 6 - C * n_mgnt['linklen'].min()
    n_mgnt['s'] = np.round(C * n_mgnt['linklen'] + D, 0)

    n_orgn = n[~n['section'].isnull()].copy()
    n_orgn['s'] = n_orgn['doc_id'].apply(lambda x: len(x))
    C = (10 - 3) / (n_orgn['s'].max() - n_orgn['s'].min())
    D = 3 - C * n_orgn['s'].min()
    n_orgn['s'] = np.round(C * n_orgn['s'] + D, 0)

    n_node_size = pd.concat([n_mgnt['s'], n_orgn['s']], axis = 0)
    n_node_size.columns = ['s']
    n = pd.concat([n, n_node_size], axis = 1)

    ## rename
    e = e.rename(columns = {'weight': 'v'})
    n = n.rename(columns = {'name': 't'})
    
    _json = make_json(n, e)
    with open("./주관,수행.json", "w") as f:
        json.dump(_json, f)
    return _json


#주관기관과 연구자 네트워크
def player_network1(PlayerEdgelist, _tree = True):
    
    PlayerEdgelist['hm_nm']=PlayerEdgelist['hm_nm']+ "("+PlayerEdgelist['prfrm_org_nm']+")"
    PlayerEdgelist['hm_nm_cd']=PlayerEdgelist['hm_nm']
    #PlayerEdgelist['hm_nm'] = PlayerEdgelist.apply(lambda x: x['hm_nm']+ "("+PlayerEdgelist['prfrm_org_nm']+")" if x['hm_nm']!=''
    #                        else x['prfrm_org_nm'], axis=1)
    
    _mgnt = PlayerEdgelist.loc[:, ['mgnt_org_cd', 'mgnt_org_nm']].rename(columns = {'mgnt_org_cd': 'code', 'mgnt_org_nm': 'name'}).drop_duplicates()
    _prfrm = PlayerEdgelist.loc[:, ['hm_nm_cd', 'hm_nm']].rename(columns = {'hm_nm_cd': 'code', 'hm_nm': 'name'}).drop_duplicates()
    _code = pd.concat([_mgnt, _prfrm]).drop_duplicates(subset='code')
    real_name_dict = {c:n for n, c in zip(_code['name'], _code['code'])}

    result = PlayerEdgelist.rename(columns = {'mgnt_org_cd': 'section', 
                                              'hm_nm_cd': 'orgn_id', 
                                              'term': 'word'})

    if not _tree:
        result_net = result.loc[:, ['orgn_id', 'section']]
        result_net = result_net.drop_duplicates().reset_index(drop = True)
        result_net = result_net.rename(columns = {'orgn_id': 'word', 
                                                  'section': 'doc_id'})

        e, n, g = df2edge(result_net, 'word', _map = False)

        return e, n, g

    e_orgn, n_orgn, _ = df2edge_player(result, 'orgn', drop_match = True, p_mode = 'max')
    e_mgnt, n_mgnt, _ = df2edge_player(result, 'mgnt', drop_match = True, p_mode = 'max')
    n_mgnt['type']=1
    n_orgn['type']=3
    
    # 수행기관과 주관기관이 둘 다 주관기관이고 같은 경우 - drop
    # 수행기관과 주관기관이 둘 다 주관기관이고 다른 경우 - keep
    e2 = pd.DataFrame({'source': result['orgn_id'],
                       'target': result['section']})
    e2 = e2.drop_duplicates(['source', 'target']).reset_index(drop = True)    

    weight = result.groupby(['section', 'orgn_id'])['doc_id'].nunique().agg(list)
    weight = pd.DataFrame(weight, columns = ['weight'])
    e2['weight'] = weight

    e = pd.concat([e2, e_mgnt], axis = 0)
    #e = e[e['source'] != e['target']]
    e = e[e['target'] != e['source']]
    e['weight'] = e['weight'] / e['weight'].max()

    n_orgn['section'] = [s[0] for s in n_orgn['section']]
    n_orgn = n_orgn.drop('id', axis = 1)

    # 수행기관 doc_id list 추가
    _doc_id_by_orgn_df = result.drop_duplicates(['orgn_id', 'doc_id'])
    _doc_id_by_orgn_df = _doc_id_by_orgn_df.groupby(['orgn_id'])['doc_id'].agg(list).reset_index().rename({'orgn_id': 'name'}, axis = 1)
    n_orgn = n_orgn.merge(_doc_id_by_orgn_df, how = 'left')

    # 주관기관 doc_id list 추가
    _doc_id_by_mgnt_orgn_df = result.drop_duplicates(['section', 'doc_id'])
    _doc_id_by_mgnt_orgn_df = _doc_id_by_mgnt_orgn_df.groupby(['section'])['doc_id'].agg(list).reset_index().rename({'section': 'name'}, axis = 1)
    n_mgnt = n_mgnt.merge(_doc_id_by_mgnt_orgn_df, how = 'left')
    n_mgnt = n_mgnt.drop('id', axis = 1)

    n = pd.concat([n_mgnt, n_orgn], axis = 0).reset_index(drop = True)
    n = n.drop_duplicates(['name']).reset_index(drop = True)    # 수행기관 역할을 하는 주관기관 drop (주관기관 node가 위에 있으므로 keep = first)
    n = n.reset_index().rename(columns = {'index': 'id'})

    # id encoding
    _dict = {i:v for i, v in zip(n['id'], n['name'])}
    e['source'] = e['source'].map({v:k for k, v in zip(_dict.keys(), _dict.values())})
    e['target'] = e['target'].map({v:k for k, v in zip(_dict.keys(), _dict.values())})
    
    n['name'] = n['name'].map(real_name_dict)

    # Node size
    n_mgnt = n[n['section'].isnull()].copy()
    C = (10 - 6) / (n_mgnt['linklen'].max() - n_mgnt['linklen'].min())
    D = 6 - C * n_mgnt['linklen'].min()
    n_mgnt['s'] = np.round(C * n_mgnt['linklen'] + D, 0)

    n_orgn = n[~n['section'].isnull()].copy()
    n_orgn['s'] = n_orgn['doc_id'].apply(lambda x: len(x))
    C = (10 - 3) / (n_orgn['s'].max() - n_orgn['s'].min())
    D = 3 - C * n_orgn['s'].min()
    n_orgn['s'] = np.round(C * n_orgn['s'] + D, 0)

    n_node_size = pd.concat([n_mgnt['s'], n_orgn['s']], axis = 0)
    n_node_size.columns = ['s']
    n = pd.concat([n, n_node_size], axis = 1)

    ## rename
    e = e.rename(columns = {'weight': 'v'})
    n = n.rename(columns = {'name': 't'})
    print(n,"\n[player_nt_node1\n")
    _json = make_json(n, e)
    with open("./주관기관,연구자.json", "w") as f:
        json.dump(_json, f)
    return _json

#주관기관,수행기관,연구자 네트워크
def player_network(PlayerEdgelist, _tree = True):
    
    
    PlayerEdgelist['hm_nm']=PlayerEdgelist['hm_nm']+ "("+PlayerEdgelist['prfrm_org_nm']+")"
    PlayerEdgelist['hm_nm_cd']=PlayerEdgelist['hm_nm']
    #PlayerEdgelist['hm_nm'] = PlayerEdgelist.apply(lambda x: x['hm_nm']+ "("+PlayerEdgelist['prfrm_org_nm']+")" if x['hm_nm']!=''
    #                        else x['prfrm_org_nm'], axis=1)
    
    _mgnt = PlayerEdgelist.loc[:, ['mgnt_org_cd', 'mgnt_org_nm']].rename(columns = {'mgnt_org_cd': 'code', 'mgnt_org_nm': 'name'}).drop_duplicates()
    _prfrm = PlayerEdgelist.loc[:, ['prfrm_org_cd', 'prfrm_org_nm']].rename(columns={'prfrm_org_cd': 'code', 'prfrm_org_nm': 'name'}).drop_duplicates()
    _rsrch = PlayerEdgelist.loc[:, ['hm_nm_cd', 'hm_nm']].rename(columns={'hm_nm_cd': 'code', 'hm_nm': 'name'}).drop_duplicates()
    _code = pd.concat([_mgnt, _prfrm,_rsrch]).drop_duplicates(subset='code')
    real_name_dict = {c:n for n, c in zip(_code['name'], _code['code'])}

    result = PlayerEdgelist.rename(columns = {'mgnt_org_cd': 'section', 
                                              'prfrm_org_cd': 'orgn_id', 
                                              'term': 'word'})
    result2 = PlayerEdgelist.rename(columns = {'prfrm_org_cd': 'section',
                                              'hm_nm_cd': 'orgn_id', 
                                              'term': 'word'})
    result['pyear'] = result['pyear'].astype('int32')
    result2['pyear'] = result2['pyear'].astype('int32')
    if not _tree:
        result_net = result2.loc[:, ['orgn_id', 'section']]
        result_net = result_net.drop_duplicates().reset_index(drop = True)
        result_net = result_net.rename(columns = {'orgn_id': 'word', 
                                                  'section': 'doc_id'})

        e, n, g = df2edge(result_net, 'word', _map = False)

        return e, n, g

    e_orgn, n_orgn, _ = df2edge_player(result, 'orgn', drop_match = True, p_mode = 'max')
    e_mgnt, n_mgnt, _ = df2edge_player(result, 'mgnt', drop_match = True, p_mode = 'max')
    e_rsrch, n_rsrch, _ = df2edge_player(result2, 'orgn', drop_match = True, p_mode = 'max')

    n_mgnt['type']=1
    n_orgn['type']=2
    n_rsrch['type']=3
    
    # 수행기관과 주관기관이 둘 다 주관기관이고 같은 경우 - drop
    # 수행기관과 주관기관이 둘 다 주관기관이고 다른 경우 - keep
    e2 = pd.DataFrame({'source': result['orgn_id'],
                       'target': result['section']})
    e2_2 = pd.DataFrame({'source': result2['orgn_id'],
                       'target': result2['section']})
    
    e2 = pd.concat([e2, e2_2], axis=0)
    #print(e2,"e2")
    e2 = e2.drop_duplicates(['source', 'target']).reset_index(drop = True)    

    weight = result.groupby(['section', 'orgn_id'])['doc_id'].nunique().agg(list)
    weight = pd.DataFrame(weight, columns = ['weight'])
    
    weight2 = result2.groupby(['section', 'orgn_id'])['doc_id'].nunique().agg(list)
    weight2 = pd.DataFrame(weight2, columns = ['weight'])
    weight = pd.concat([weight, weight2], axis=0).reset_index(drop = True) 
    e2['weight'] = weight

    e = pd.concat([e2, e_mgnt], axis = 0).reset_index(drop = True)
    e = e[e['target'] != e['source']]
    e['weight'] = e['weight'] / e['weight'].max()

    n_rsrch['section'] = [s[0] for s in n_rsrch['section']]
    n_rsrch = n_rsrch.drop('id', axis = 1)
    
    n_orgn['section'] = [s[0] for s in n_orgn['section']]
    n_orgn = n_orgn.drop('id', axis = 1)
    
    # 수행기관 doc_id list 추가
    _doc_id_by_orgn_df = result.drop_duplicates(['orgn_id', 'doc_id'])
    _doc_id_by_orgn_df = _doc_id_by_orgn_df.groupby(['orgn_id'])['doc_id'].agg(list).reset_index().rename({'orgn_id': 'name'}, axis = 1)
    n_orgn = n_orgn.merge(_doc_id_by_orgn_df, how = 'left')

    _doc_id_by_rsrch_df = result2.drop_duplicates(['orgn_id', 'doc_id'])
    _doc_id_by_rsrch_df = _doc_id_by_rsrch_df.groupby(['orgn_id'])['doc_id'].agg(list).reset_index().rename({'orgn_id': 'name'}, axis = 1)
    n_rsrch = n_rsrch.merge(_doc_id_by_rsrch_df, how = 'left')
    
    
    # 주관기관 doc_id list 추가
    _doc_id_by_mgnt_orgn_df = result.drop_duplicates(['section', 'doc_id'])
    _doc_id_by_mgnt_orgn_df = _doc_id_by_mgnt_orgn_df.groupby(['section'])['doc_id'].agg(list).reset_index().rename({'section': 'name'}, axis = 1)
    n_mgnt = n_mgnt.merge(_doc_id_by_mgnt_orgn_df, how = 'left')
    n_mgnt = n_mgnt.drop('id', axis = 1)


    n = pd.concat([n_mgnt, n_orgn,n_rsrch], axis = 0).reset_index(drop = True)
    n = n.drop_duplicates(['name']).reset_index(drop = True)    # 수행기관 역할을 하는 주관기관 drop (주관기관 node가 위에 있으므로 keep = first)
    n = n.reset_index().rename(columns = {'index': 'id'})
    print(n)
    # id encoding
    _dict = {i:v for i, v in zip(n['id'], n['name'])}
    e['source'] = e['source'].map({v:k for k, v in zip(_dict.keys(), _dict.values())})
    e['target'] = e['target'].map({v:k for k, v in zip(_dict.keys(), _dict.values())})
    e=e[~e['target'].isna()]
    n['name'] = n['name'].map(real_name_dict)

    # Node size
    n_mgnt = n[n['section'].isnull()].copy()
    C = (10 - 6) / (n_mgnt['linklen'].max() - n_mgnt['linklen'].min())
    D = 6 - C * n_mgnt['linklen'].min()
    n_mgnt['s'] = np.round(C * n_mgnt['linklen'] + D, 0)


    n_orgn = pd.concat([n_orgn, n_rsrch]).reset_index(drop = True)
    n_orgn = n[~n['section'].isnull()].copy()
    n_orgn['s'] = n_orgn['doc_id'].apply(lambda x: len(x))
    C = (10 - 3) / (n_orgn['s'].max() - n_orgn['s'].min())
    D = 3 - C * n_orgn['s'].min()
    n_orgn['s'] = np.round(C * n_orgn['s'] + D, 0)
    

    n_node_size = pd.concat([n_mgnt['s'], n_orgn['s']], axis = 0)
    n_node_size.columns = ['s']
    n = pd.concat([n, n_node_size], axis = 1)
    
    ## rename
    e = e.rename(columns = {'weight': 'v'})
    n = n.rename(columns = {'name': 't'})
    print(n,"\n[player_nt_node2\n")
    #e.to_csv('mul_net_edge.csv', encoding='utf-8-sig')
    #n.to_csv('mul_net_node.csv', encoding='utf-8-sig')
    _json = make_json(n, e)
    with open("./주관,수행,연구자.json", "w") as f:
        json.dump(_json, f)
    return _json

#기술분류와 연구자 네트워크
def player_network3(PlayerEdgelist, _tree = True):
    PlayerEdgelist['doc_section_name']=PlayerEdgelist['doc_section']
    PlayerEdgelist['hm_nm']=PlayerEdgelist['hm_nm']+ "("+PlayerEdgelist['prfrm_org_nm']+")"
    PlayerEdgelist['hm_nm_cd']=PlayerEdgelist['hm_nm']
    #PlayerEdgelist['hm_nm'] = PlayerEdgelist.apply(lambda x: x['hm_nm']+ "("+PlayerEdgelist['prfrm_org_nm']+")" if x['hm_nm']!=''
    #                        else x['prfrm_org_nm'], axis=1)
    
    _mgnt = PlayerEdgelist.loc[:, ['doc_section', 'doc_section_name']].rename(columns={'doc_section': 'code', 'doc_section_name': 'name'}).drop_duplicates()
    _prfrm = PlayerEdgelist.loc[:, ['hm_nm_cd', 'hm_nm']].rename(columns = {'hm_nm_cd': 'code', 'hm_nm': 'name'}).drop_duplicates()
    _code = pd.concat([_mgnt, _prfrm]).drop_duplicates(subset='code')
    real_name_dict = {c:n for n, c in zip(_code['name'], _code['code'])}

    result = PlayerEdgelist.rename(columns = {'doc_section': 'section', 
                                              'hm_nm_cd': 'orgn_id', 
                                              'term': 'word'})
    result['pyear'] = result['pyear'].astype('int32')
    if not _tree:
        result_net = result.loc[:, ['orgn_id', 'section']]
        result_net = result_net.drop_duplicates().reset_index(drop = True)
        result_net = result_net.rename(columns = {'orgn_id': 'word', 
                                                  'section': 'doc_id'})

        e, n, g = df2edge(result_net, 'word', _map = False)

        return e, n, g

    e_orgn, n_orgn, _ = df2edge_player(result, 'orgn', drop_match = True, p_mode = 'max')
    e_mgnt, n_mgnt, _ = df2edge_player(result, 'mgnt', drop_match = True, p_mode = 'max')
    n_mgnt['type']=4
    n_orgn['type']=3
    # 수행기관과 주관기관이 둘 다 주관기관이고 같은 경우 - drop
    # 수행기관과 주관기관이 둘 다 주관기관이고 다른 경우 - keep
    e2 = pd.DataFrame({'source': result['orgn_id'],
                       'target': result['section']})
    e2 = e2.drop_duplicates(['source', 'target']).reset_index(drop = True)    

    weight = result.groupby(['section', 'orgn_id'])['doc_id'].nunique().agg(list)
    weight = pd.DataFrame(weight, columns = ['weight'])
    e2['weight'] = weight

    e = pd.concat([e2], axis = 0)
    #e = e[e['source'] != e['target']]
    e = e[e['target'] != e['source']]
    e['weight'] = e['weight'] / e['weight'].max()

    n_orgn['section'] = [s[0] for s in n_orgn['section']]
    n_orgn = n_orgn.drop('id', axis = 1)

    # 수행기관 doc_id list 추가
    _doc_id_by_orgn_df = result.drop_duplicates(['orgn_id', 'doc_id'])
    _doc_id_by_orgn_df = _doc_id_by_orgn_df.groupby(['orgn_id'])['doc_id'].agg(list).reset_index().rename({'orgn_id': 'name'}, axis = 1)
    n_orgn = n_orgn.merge(_doc_id_by_orgn_df, how = 'left')

    # 주관기관 doc_id list 추가
    _doc_id_by_mgnt_orgn_df = result.drop_duplicates(['section', 'doc_id'])
    _doc_id_by_mgnt_orgn_df = _doc_id_by_mgnt_orgn_df.groupby(['section'])['doc_id'].agg(list).reset_index().rename({'section': 'name'}, axis = 1)
    n_mgnt = n_mgnt.merge(_doc_id_by_mgnt_orgn_df, how = 'left')
    n_mgnt = n_mgnt.drop('id', axis = 1)

    n = pd.concat([n_mgnt, n_orgn], axis = 0).reset_index(drop = True)
    n = n.drop_duplicates(['name']).reset_index(drop = True)    # 수행기관 역할을 하는 주관기관 drop (주관기관 node가 위에 있으므로 keep = first)
    n = n.reset_index().rename(columns = {'index': 'id'})

    # id encoding
    _dict = {i:v for i, v in zip(n['id'], n['name'])}
    e['source'] = e['source'].map({v:k for k, v in zip(_dict.keys(), _dict.values())})
    e['target'] = e['target'].map({v:k for k, v in zip(_dict.keys(), _dict.values())})
    
    n['name'] = n['name'].map(real_name_dict)

    # Node size
    n_mgnt = n[n['section'].isnull()].copy()
    C = (10 - 6) / (n_mgnt['linklen'].max() - n_mgnt['linklen'].min())
    D = 6 - C * n_mgnt['linklen'].min()
    n_mgnt['s'] = np.round(C * n_mgnt['linklen'] + D, 0)

    n_orgn = n[~n['section'].isnull()].copy()
    n_orgn['s'] = n_orgn['doc_id'].apply(lambda x: len(x))
    C = (10 - 3) / (n_orgn['s'].max() - n_orgn['s'].min())
    D = 3 - C * n_orgn['s'].min()
    n_orgn['s'] = np.round(C * n_orgn['s'] + D, 0)

    n_node_size = pd.concat([n_mgnt['s'], n_orgn['s']], axis = 0)
    n_node_size.columns = ['s']
    n = pd.concat([n, n_node_size], axis = 1)

    ## rename
    e = e.rename(columns = {'weight': 'v'})
    n = n.rename(columns = {'name': 't'})
    print(n,"\n[player_nt_node3\n")
    #e.to_csv('player_net_edge.csv', encoding='utf-8-sig')
    #n.to_csv('player_net_node.csv', encoding='utf-8-sig')
    _json = make_json(n, e)
    with open("./분류기술.json", "w") as f:
        json.dump(_json, f)
    return _json


def extract_topic_word(EgoEdgelist, ResizeEdgelist, s_dict, window_size = 15, alpha = 0.0):
    
    EgoEdgelist = EgoEdgelist.reset_index(drop=True)
    dfreq = EgoEdgelist.groupby('word').size().rename('dfreq') # 검색된 문서 내 dfreq를 새롭게 계산
    EgoEdgelist = pd.merge(EgoEdgelist, dfreq, left_on = 'word', right_index = True)
    EgoEdgelist = EgoEdgelist[EgoEdgelist['dfreq'] > 1] # 최소 2개 이상의 문서에 등장하는 단어만 사용하도록 조정, 1개 문서 내에만 존재하는 강한 연결을 만들지 않도록 함

    if EgoEdgelist.empty:
        return None, None, None
    
    ##################################

    if not s_dict:
        s_dict = {i:0 for i in ResizeEdgelist['doc_id'].unique()} 

    ResizeEdgelist_paper = ResizeEdgelist.copy()

    select_doc = list(itertools.islice(s_dict.keys(),300))
    ResizeEdgelist = ResizeEdgelist[ResizeEdgelist.doc_id.isin(select_doc)]

    ##################################

    _data_c = ResizeEdgelist[ResizeEdgelist.col_type == 'compound'].copy()
    
    if _data_c.empty:
        pmi_mat, p_rc, w_dict = make_pmi(ResizeEdgelist, window_size = window_size, alpha = alpha, sample_n = False, min_pmi = 0)
        p_rc = p_rc.tocsr()

        p_rc.data = -np.log(p_rc.data)
        pmi_mat_s = sparse.dok_matrix(pmi_mat.shape) 
        pmi_mat_s[p_rc.nonzero()] = pmi_mat[p_rc.nonzero()] / p_rc[p_rc.nonzero()] # pmi 행렬 표준화, pmi값을 ln(p_rc)로 나눈 값이다. (ln : 자연로그)
        pmi_mat = pmi_mat_s.tocsr()

        topic_word, word_json = word_network(ResizeEdgelist, pmi_mat, w_dict)
        paper_json, paper_cent = paper_network(ResizeEdgelist, EgoEdgelist, topic_word, s_dict, _revert = 1)
        
    else:
        _data_c.loc[:,'component'] = _data_c.word.apply(lambda x:x.split(' '))
        _data_nc = ResizeEdgelist[ResizeEdgelist.col_type != 'compound']
        pmi_mat, p_rc, w_dict = make_pmi(_data_nc, window_size = window_size, alpha = alpha, sample_n = False, min_pmi = 0)
        p_rc = p_rc.tocsr()

        p_rc.data = -np.log(p_rc.data)
        pmi_mat_s = sparse.dok_matrix(pmi_mat.shape) 
        pmi_mat_s[p_rc.nonzero()] = pmi_mat[p_rc.nonzero()] / p_rc[p_rc.nonzero()] # pmi 행렬 표준화, pmi값을 ln(p_rc)로 나눈 값이다. (ln : 자연로그)
        pmi_mat = pmi_mat_s.tocsr()

        c_idx = _data_c.apply(lambda x: [w_dict[i] for i in x.component], axis=1).tolist()
        _cg = pd.Series([list(itertools.combinations(i, 2)) for i in c_idx])
        _cg_n = [len(c) for c in _cg]

        list_1d = list(itertools.chain(*_cg))
        list_1d_r = [l[0] for l in list_1d]
        list_1d_c = [l[1] for l in list_1d]
        list_1d_v = iter(pmi_mat[list_1d_r, list_1d_c].tolist()[0])
        _cg = [np.array(list((itertools.islice(list_1d_v,i)))).mean() for i in _cg_n]

        _data_c.loc[:, 'mpmi'] = _cg
        # _data_c.loc[:,'mpmi'] = _cg.apply(lambda x:np.array([pmi_mat[i] for i in x]).mean()).tolist()

        _data_c = _data_c[_data_c.mpmi > 0.5] # 좋은 합성어 고르기

        if _data_c.empty:
            ResizeEdgelist = _data_nc
        
        else:
            _data_pmi = _data_c.groupby('doc_id').apply(lambda x:x.sort_values('mpmi').head(5)).reset_index(drop = True)
            _data_pmi = _data_pmi.loc[:,_data_nc.columns]
            ResizeEdgelist = pd.concat([_data_nc,_data_pmi])    

        topic_word, word_json = word_network(ResizeEdgelist, pmi_mat, w_dict)        
        paper_json, paper_cent = paper_network(ResizeEdgelist_paper, EgoEdgelist, topic_word, s_dict, _revert = 1)        

    return word_json, paper_json, paper_cent


#### 함수 적용 순서 ####

# rawdata_class = make_rawdata_class(rawdata) # rawdata : 전체 문서에 대한 edgelist, make_rawdata_class는 매번 실행하지 않아도 되는 함수
# middle_all, topic_all = make_base_dict(rawdata_class)
# bridge_word = extract_bridge_word(EgoEdgelist, middle_all) # EgoEdgelist : 문서검색을 통해 축소된 edgelist
# topic_word, _json = extract_topic_word(EgoEdgelist, topic_all)


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
    #pd.set_option('display.max_columns', None)
    #pd.set_option('display.max_rows', None)
   
    start_time = timeit.default_timer()
    
    target_db = 'kistep_sbjt'
    we_edge = SnaEdgelist()
    p_edge = PlayerEdgelist()
    doc_size = 760
    es_query = ElasticQuery(target_db)

    s_list = '인공지능'
    search_result =es_query.get_docs_by_search_text(search_text=s_list,size=doc_size,fields=['*'])
    doc_ids_score = {x['_id']:x['_score'] for x in search_result}
    doc_ids = list(doc_ids_score.keys())
    doc_size = len(doc_ids)
    # _doc = pd.read_parquet('./compound_df.parquet')
    # _ids = _doc.doc_id.unique().tolist()
    # _size = len(_ids)
    # # all_edgelist = we_edge.get_all_edgelist(target_db=target_db)
    a_time=timeit.default_timer()
    ego_edgelist, resize_ego_edgelist = we_edge.get_sna_refined_ego_edgelist(target_db=target_db, doc_ids=doc_ids, size=doc_size)        
    print(ego_edgelist,"ego_edgelist\n")
    print(ego_edgelist.columns,"ego_edgelist.columns\n")
    print(resize_ego_edgelist,"resize_ego_edgelist\n")
    print(resize_ego_edgelist.columns,"resize_ego_edgelist.columns\n")
    
    
    # # # two-mode edgelist를 축소, paper network에만 적용할지, 전체 edgelist를 적용할지?
    # # dfreq = refined_ego_edgelist.groupby(['word']).size().rename('dfreq')
    # # refined_ego_edgelist = pd.merge(refined_ego_edgelist, dfreq, left_on = 'word', right_index = True)
    # # g = df2edge(refined_ego_edgelist, 'paper')[2]
    # # N = refined_ego_edgelist.doc_id.unique().size
    # # refined_ego_edgelist['bm25'] = bm25(N,refined_ego_edgelist.dtfreq, refined_ego_edgelist.dfreq).rename('bm25')
    # # term_indice_df = ego_edgelist_node_controll(refined_ego_edgelist, g, filtering_dict)
    # # reduce_edgelist = complete_egoedgelist(refined_ego_edgelist, term_indice_df, filtering_dict)

    Edgelist_class = make_rawdata_class(all_edgelist)
    # # middle_all, topic_all, stop_all = make_base_dict(Edgelist_class, qs=0.001, nm=30)    
    # # bridge_word = extract_bridge_word(ego_edgelist, middle_all)
    # paper_json = ''
    # # print(ego_edgelist, resize_ego_edgelist)

    if not ego_edgelist.empty and not resize_ego_edgelist.empty:
        
        word_json, paper_json, paper_cent = extract_topic_word(ego_edgelist, resize_ego_edgelist, doc_ids_score)
        b_time=timeit.default_timer()
        print(" ttt: ", b_time - a_time)
        print("sna edgelist: ", timeit.default_timer() - start_time)
    # with open("./word_test.json", "w") as f:
    #     json.dump(word_json, f)
    
    ego_player, refined_ego_player = p_edge.get_player_refined_ego_edgelist(target_db, doc_ids, size=doc_size, offsets=False, positions=False, return_original=True)

    if not refined_ego_player.empty:
        player_json = orgn_network(refined_ego_player)
        #player_json = player_network1(refined_ego_player)
        #player_json = player_network2(refined_ego_player)
        #player_json = player_network(refined_ego_player)
        print("player edgelist: ", timeit.default_timer() -(b_time-a_time)- start_time)
        print("sna,player time : ", timeit.default_timer() - start_time)
        

#df1------------------------
    start_time = timeit.default_timer()
    data = pd.read_csv('./score_filtered_90_matrix_test.csv',encoding='cp949')
    
    #print(data.loc[:,['doc_id', 'cnn']])
    data.fillna(0, inplace=True)
    #print(data.loc[:,])
    #data_short=copy.copy(data[0:150])
    
    data_mat=data.drop(columns = ['doc_id'])
    data_mat_test=copy.copy(data_mat)
    
    _max=data_mat.max().max()
    _min=data_mat[data_mat!=0].min().min()

    w_dict = {i:v for i,v in enumerate(data_mat.columns)}
    p_dict = {i:v for i,v in enumerate(data.doc_id.unique())}
    dic = defaultdict(float)
    
    for i,(columnName, columnData) in enumerate(data_mat.iteritems()):
        #print(i,columnName,columnData)
        for j,v in enumerate(columnData):
            if v>0:
                dic[j]+=v

    s_dic=dict(zip(p_dict.values(),dic.values()))
    
    #df1 전처리
    for (columnName, columnData) in data_mat.iteritems():
        data_mat_test[columnName]=data_mat[columnName].apply(lambda x: (x - _min) / (_max - _min) if x>0.0 else 0)
    #data_mat_test = pd.concat([data['doc_id'], data_mat],axis = 1)
    #print(f'\n{data_mat}\n')

    src=[]
    dst=[]
    weight=[]
   
    df1 = pd.DataFrame(columns=ego_edgelist.columns)

    for i,(columnName, columnData) in enumerate(data_mat.iteritems()):
        #print(i,columnName,columnData)
        for j,v in enumerate(columnData):
            if v>0:
                #print(i,j,columnName,v)
                src.append(j)
                dst.append(columnName)
                weight.append(v)
    
    df1['doc_id']=src
    df1['word']=dst
    df1['bm25']=weight
    df1['doc_id']=df1['doc_id'].map(p_dict)
    df1['col_type']='analysis_target_text'
    df1['title']=df1['doc_id']

    list1=[]
    list2=[]
    for i,v in enumerate(df1.iterrows()):
        list1.append(['2020','2021'])
        list2.append(['UK'])
    df1['pyear']=list1
    df1['section']=list2


    #make_pmi(df1)
    data_mat=data_mat.to_numpy()
    # one_mat = np.dot(data_mat.T, data_mat)
    # one_mat = sparse.csr_matrix(one_mat)
    one_mat_test = np.dot(data_mat_test.T, data_mat_test)
    one_mat_test = sparse.csr_matrix(one_mat_test)
    w_dict_inv = {v:i for i, v in zip(w_dict.keys(), w_dict.values())}
    

    topic_word, word_json = word_network_bm(df1, one_mat_test, w_dict_inv)
    paper_json, paper_cent = paper_network_bm_new(df1, df1, topic_word, s_dic, _revert = 1)
    print("bm25_sna edgelist: ", timeit.default_timer() - start_time)
    # with open("./paper_short_test.json", "w") as f:
    #     json.dump(paper_json2, f)
    '''
#df2--------------------

    data2 = pd.read_csv('./score_filtered_90_matrix_test2.csv',encoding='cp949')
    data2.fillna(0, inplace=True)
    data_mat_short=data2.drop(columns = ['doc_id'])
    data_mat_test_short=copy.copy(data_mat_short)

    _max_short=data_mat_short.max().max()
    _min_short=data_mat_short[data_mat_short!=0].min().min()

    w_dict_short = {i:v for i,v in enumerate(data_mat_short.columns)}
    p_dict_short = {i:v for i,v in enumerate(data2.doc_id.unique())}
    dic_short = defaultdict(float)
    
    for i,(columnName, columnData) in enumerate(data_mat_short.iteritems()):
        #print(i,columnName,columnData)
        for j,v in enumerate(columnData):
            if v>0:
                dic_short[j]+=v

    s_dic_short=dict(zip(p_dict_short.values(),dic_short.values()))

    for (columnName, columnData) in data_mat_short.iteritems():
        data_mat_test_short[columnName]=data_mat_short[columnName].apply(lambda x: (x - _min_short) / (_max_short - _min_short) if x>0.0 else 0)
        #data_mat_test_short[columnName]=data_mat_short[columnName].apply(lambda x: 1 if x>0.0 else 0)
    #data_mat_test = pd.concat([data['doc_id'], data_mat],axis = 1)
    #print(f'\n{data_mat}\n')

    src=[]
    dst=[]
    weight=[]
   
    df2 = pd.DataFrame(columns=ego_edgelist.columns)

    for i,(columnName, columnData) in enumerate(data_mat_short.iteritems()):
        #print(i,columnName,columnData)
        for j,v in enumerate(columnData):
            if v>0:
                #print(i,j,columnName,v)
                src.append(j)
                dst.append(columnName)
                weight.append(v)
    
    df2['doc_id']=src
    df2['word']=dst
    df2['bm25']=weight
    
    df2['doc_id']=df2['doc_id'].map(p_dict)
    df2['col_type']='analysis_target_text'
    df2['title']=df2['doc_id']
    list1=[]
    list2=[]
    for i,v in enumerate(df2.iterrows()):
        list1.append(['2020','2021'])
        list2.append(['UK'])
    df2['pyear']=list1
    df2['section']=list2

    #make_pmi(df2)
    data_mat_short=data_mat_short.to_numpy()
    # one_mat = np.dot(data_mat.T, data_mat)
    # one_mat = sparse.csr_matrix(one_mat)
    one_mat_test_short = np.dot(data_mat_test_short.T, data_mat_test_short)
    one_mat_test_short = sparse.csr_matrix(one_mat_test_short)
    w_dict_inv_short = {v:i for i, v in zip(w_dict_short.keys(), w_dict_short.values())}
    
    
    topic_word2, word_json2 = word_network_bm(df2, one_mat_test_short, w_dict_inv_short)
    # with open("./word_test.json_not_force", "w") as f:
    #     json.dump(word_json, f)

    #paper_json, paper_cent = paper_network_bm(df1, df1, topic_word, s_dic, _revert = 1)
    # paper_json, paper_cent = paper_network_bm_new(df1, df1, topic_word, s_dic, _revert = 1)
    # with open("./paper_test.json", "w") as f:
    #     json.dump(paper_json, f)
    paper_json2, paper_cent2 = paper_network_bm_new(df2, df2, topic_word2, s_dic_short, _revert = 1)
    # with open("./paper_short_test.json", "w") as f:
    #     json.dump(paper_json2, f)
    '''
#mode2(doc,word)------------------------------------------------------
    b_time = timeit.default_timer()
    
    data = pd.read_csv('./score_filtered_90_matrix_test.csv',encoding='cp949')
    data.fillna(0, inplace=True)
    data_mat_orgn=copy.copy(data_mat)
    data_mat=data.drop(columns = ['doc_id'])
    p_dict = {i:v for i,v in enumerate(data.doc_id.unique())}

    for (columnName, columnData) in data_mat.iteritems():
        data_mat_test[columnName]=data_mat[columnName].apply(lambda x: 1 if x>0.0 else 0)
    src=[]
    dst=[]
    weight=[]
    df3 = pd.DataFrame(columns=refined_ego_player.columns)
    for i,(columnName, columnData) in enumerate(data_mat.iteritems()):
        #print(i,columnName,columnData)
        for j,v in enumerate(columnData):
            if v>0:
                #print(i,j,columnName,v)
                src.append(j)
                dst.append(columnName)
                weight.append(v)
    df3['doc_id']=src
    df3['term']=dst
    df3['doc_id']=df3['doc_id'].map(p_dict)
    df3['title']=df3['mgnt_org_cd']=df3['mgnt_org_nm']=df3['term']
    df3['prfrm_org_cd']=df3['prfrm_org_nm']=df3['doc_id']

    player_json=doc_word_bm(df3)
    print("bm25_player : ", timeit.default_timer() - b_time)
    print("bm25_sna,player edgelist: ", timeit.default_timer() - start_time)
    with open("./player_bm25.json", "w") as f:
        json.dump(player_json, f)