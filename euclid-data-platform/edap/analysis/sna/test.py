import pandas as pd
import numpy as np
import json
from copy import copy
import igraph
from collections import Counter, defaultdict
import pandas as pd
import numpy as np
import itertools
from functools import reduce
from scipy import sparse

# for soynlp pmi
from scipy.sparse import diags
from scipy.sparse import csr_matrix

#pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)
def test(x):
    if x>0: return ((x - _min) / (_max - _min))
    else : return 0
        
        
data = pd.read_csv('C:/jupyter-notebook/KIS_LOCAL/kistep-lime-net/euclid-data-platform/edap/analysis/sna/score_filtered_90_matrix_test.csv',encoding='cp949')
#print(data.loc[:,['doc_id', 'cnn']])
data.fillna(0, inplace=True)
#print(data.loc[:,])

data_mat=data.drop(columns = ['doc_id'])
_max=data_mat.max().max()
_min=data_mat[data_mat!=0].min().min()

w_dict = {i:v for i,v in enumerate(data_mat.columns)}
p_dict = {i:v for i,v in enumerate(data.doc_id.unique())}

for (columnName, columnData) in data_mat.iteritems():
    data_mat[columnName]=data_mat[columnName].apply(lambda x: (x - _min) / (_max - _min) if x>0.0 else 0)
#print(f'\n{data_mat}\n')


#data_mat = pd.concat([data['doc_id'], data_mat],axis = 1)
data_mat=data_mat.to_numpy()
s_two_mat = sparse.csr_matrix(data_mat)


def make_one_graph(one_node, one_edge):       

    edgelist = one_edge[['source', 'target']]
    
    # tolist에서 시간 소요되는데, 더 이상 줄일 방법은 없음
    EdgeTuple = edgelist.to_records(index=False).tolist()
    
    # Graph 생성
    graph = igraph.Graph()
    graph.add_vertices(len(one_node))
    graph.add_edges(EdgeTuple)

    # Graph에 Node 정보 입력
    graph.vs["name"] = one_node.name.tolist()

    # Graph에 Edge의 정보 입력
    graph.es["weight"] = one_edge.weight.tolist()

    return graph

def make_graph_value(one_mat, _dict, p_mode):
        
    _n,_g = sparse.csgraph.connected_components(one_mat)
    one_mat_deg = pd.DataFrame(one_mat.diagonal(),_dict.values())[0].to_dict()
    one_mat = sparse.triu(one_mat,k=1, format='csr')
    source, target = one_mat.nonzero()
#             source, target = np.nonzero(one_mat>0) 
    weight = one_mat.data            

    one_edge = pd.DataFrame({'source':source, 'target':target, 'weight':weight})                
    one_node = pd.DataFrame({'id':_dict.keys(),'name':_dict.values(), 'cluster':_g, 'linklen':one_mat_deg.values()})                           
        
    if p_mode == 'max':
            one_graph = 1
        
    elif p_mode == 'default':
        one_graph = make_one_graph(one_node, one_edge)
        to_delete_ids = [v.index for v in one_graph.vs if v.degree() == 0]
        one_graph.delete_vertices(to_delete_ids)
        
    one_edge.source = one_edge.source.map(_dict)
    one_edge.target = one_edge.target.map(_dict)
        
    return one_edge, one_node, one_graph        


word_one_mat = np.dot(s_two_mat.T,s_two_mat)


e_reduce, n_reduce, w_graph = make_graph_value(word_one_mat, w_dict, p_mode = 'default')
#p_edge, p_node, p_graph = make_graph_value(paper_one_mat, w_dict, p_mode = 'default')
print(e_reduce,"e_reduce\n")
print(n_reduce,"n_reduce\n")


'''
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

e_reduce = e_reduce.rename(columns = {'weight': 'v'})   
n_reduce = n_reduce.rename(columns = {'name': 't'})    
_dict = {i:v for i, v in zip(n_reduce['id'], n_reduce['t'])}
e_reduce['source'] = e_reduce['source'].map({v:i for i, v in zip(_dict.keys(), _dict.values())})
e_reduce['target'] = e_reduce['target'].map({v:i for i, v in zip(_dict.keys(), _dict.values())})
n_reduce['pyear'] = "2020"
n_reduce['section'] = None
n_reduce['type']=1
n_reduce['doc_id']=None
n_reduce['s'] = 7

_json = make_json(n_reduce, e_reduce)
with open("./test.json", "w") as f:
        json.dump(_json, f)
'''