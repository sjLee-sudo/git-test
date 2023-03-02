import pandas as pd
import numpy as np
import itertools
import igraph
import copy
from scipy import sparse

from collections import Counter, defaultdict

# for soynlp pmi
from scipy.sparse import diags
from scipy.sparse import csr_matrix

def make_pmi(_data, window_size=1, alpha = 0, min_pmi = 0, sample_n = False, seed=10):       
    
    data = copy.copy(_data)
    
    if sample_n:        
        word_sample = data.word.sample(n=sample_n, random_state = seed)
        data = data[data.word.isin(word_sample)]
    
    w_dict = {v:i for i,v in enumerate(data.word.unique())}
    data.word = data.word.map(w_dict)

    data_word = list(itertools.chain.from_iterable(itertools.repeat(word,len(n)) for (word,n) in zip(data.word, data.position)))
    data_paper = list(itertools.chain.from_iterable(itertools.repeat(paper,len(n)) for (paper,n) in zip(data.doc_id, data.position)))
    data_idx = list(itertools.chain(*data.position))

    data_raw = pd.DataFrame({'word':data_word,'doc_id':data_paper,'position':data_idx})
    ####################################################
    # 해결이 필요한 코드 : 성능향상을 위해
#     data_raw = data_raw.sort_values('doc_id')
#     result = data_raw.groupby('doc_id').progress_apply(lambda x: doc2pmi(x,window_size))
    
    data_raw = data_raw.sort_values('doc_id')
    result = data_raw.set_index('doc_id')
    result = [(x, window_size) for i, x in result.groupby(level=0, sort=False)]
    result = list(itertools.starmap(doc2pmi, result))
    ####################################################
    
#     result = list(itertools.chain(*result))
#     result = pd.DataFrame({'word':data_raw.word,'co_word':result})
#     result = result.groupby('word').agg({'co_word': 'sum'})
#     result = result.reset_index()
    
    result = list(itertools.chain(*result))
    result = pd.DataFrame({'word':data_raw.word,'co_word':result})
    
    # method 1
    # groups = defaultdict(lambda: [])
    # for _id, word in result.values:
    #     groups[_id].append(word)
    # co_word = [list(itertools.chain(*groups[i])) for i in range(len(w_dict))]
    # co_word = [[i for i in _list if i != w] for w,_list in enumerate(co_word)]
    # result = pd.DataFrame({'word':list(range(len(w_dict))), 'co_word':co_word})

    # method 2
    result_v = iter(result.sort_values('word').co_word.tolist())
    result_n = result.groupby('word').size()
    result = [sorted(list(itertools.chain(*(list(itertools.islice(result_v,i)))))) for i in result_n]
    co_word = [[i for i in _list if i != w] for w,_list in enumerate(result)]
    result = pd.DataFrame({'word':list(range(len(w_dict))), 'co_word':co_word})
    
    _idx = list(itertools.chain.from_iterable(itertools.repeat(word,len(n)) for (word,n) in zip(result.word, result.co_word)))
    _cw = list(itertools.chain(*result.co_word))
    
    data_sparse = pd.DataFrame({'idx':_idx,'cw':_cw})
    data_sparse['value'] = 1
    
    # sparse matrix를 추가할 때, 중복되는 index가 들어가면 +1이 됨
    # 단순 중복인지, 의미가 있는 값인지 판별할 필요성
    
    # data_sparse = data_sparse.drop_duplicates()
    
    m = sparse.coo_matrix((data_sparse.value, (data_sparse.idx, data_sparse.cw)))    
    
    pmi_mat,p_r,p_c, p_rc = pmi(m, alpha = alpha, min_pmi=min_pmi)

    return pmi_mat, p_rc, w_dict

def doc2pmi(_pf, window_size):
    words = []
    max_v = max(_pf.position)
    _pfv = _pf.values

    for v in _pf.position:
        word = []             
       
        if (v - window_size) <= 0:        
            lower = 0
            upper = v + window_size        
            word.append(_pfv[:,0][(_pfv[:,1]>=lower)&(_pfv[:,1]<=upper)&(_pfv[:,1] != v)])
            
        elif (v + window_size) > max_v:
            lower = v - window_size
            upper = max_v
            word.append(_pfv[:,0][(_pfv[:,1]>=lower)&(_pfv[:,1]<=upper)&(_pfv[:,1] != v)])

        else:           
            lower = v - window_size
            upper = v + window_size
            word.append(_pfv[:,0][(_pfv[:,1]>=lower)&(_pfv[:,1]<=upper)&(_pfv[:,1] != v)])

        word = list(set(list(itertools.chain(*word))))
        words.append(word)
    
    return words

# def doc2pmi(_pf, window_size):
#     words = []
#     max_v = max(_pf.position)
#     _pfv = _pf.values

#     for v in _pf.position:
#         word = []             

#         # backward
#         if (v - window_size) <= 0:        
#             lower = 0
#             upper = v        
#             word.append(_pfv[:,0][(_pfv[:,2]>=lower)&(_pfv[:,2]<=upper)&(_pfv[:,2] != v)])

#         else:
#             lower = v - window_size
#             upper = v
#             word.append(_pfv[:,0][(_pfv[:,2]>=lower)&(_pfv[:,2]<=upper)&(_pfv[:,2] != v)])

#         # forward
#         if (v + window_size) > max_v:
#             lower = v
#             upper = max_v
#             word.append(_pfv[:,0][(_pfv[:,2]>=lower)&(_pfv[:,2]<=upper)&(_pfv[:,2] != v)])

#         else:           
#             lower = v
#             upper = v + window_size
#             word.append(_pfv[:,0][(_pfv[:,2]>=lower)&(_pfv[:,2]<=upper)&(_pfv[:,2] != v)])

#         word = list(set(list(itertools.chain(*word))))
#         words.append(word)
    
#     return words


def _logarithm_and_ppmi(exp_pmi, min_exp_pmi):
    n, m = exp_pmi.shape

    # because exp_pmi is sparse matrix and type of exp_pmi.data is numpy.ndarray
    rows, cols = exp_pmi.nonzero()
    data = exp_pmi.data

    indices = np.where(data >= min_exp_pmi)[0]
    rows = rows[indices]
    cols = cols[indices]
    data = data[indices]

    # apply logarithm
    data = np.log(data)

    # new matrix
    exp_pmi_ = csr_matrix((data, (rows, cols)), shape=(n, m))
    return exp_pmi_


def _as_diag(px, alpha):
    px_diag = diags(px.tolist()[0])
    px_diag.data[0] = np.asarray([0 if v == 0 else 1 / (v + alpha) for v in px_diag.data[0]])
    return px_diag


def pmi(X, py=None, min_pmi=0, alpha=0.0, beta=1):
    
    """
    :param X: scipy.sparse.csr_matrix
        (word, contexts) sparse matrix
    :param py: numpy.ndarray
        (1, word) shape, probability of context words.
    :param min_pmi: float
        Minimum value of pmi. all the values that smaller than min_pmi
        are reset to zero.
        Default is zero.
    :param alpha: float
        Smoothing factor. pmi(x,y; alpha) = p_xy /(p_x * (p_y + alpha))
        Default is 0.0
    :param beta: float
        Smoothing factor. pmi(x,y) = log ( Pxy / (Px x Py^beta) )
        Default is 1.0
    Returns
    ----------
    pmi : scipy.sparse.csr_matrix
        (word, contexts) pmi value sparse matrix
    px : numpy.ndarray
        Probability of rows (items)
    py : numpy.ndarray
        Probability of columns (features)
    Usage
    -----
        >>> pmi, px, py = pmi_memory_friendly(X, py=None, min_pmi=0, alpha=0, beta=1.0)
    """

    assert 0 < beta <= 1

    # convert x to probability matrix & marginal probability
    px = np.asarray((X.sum(axis=1) / X.sum()).reshape(-1))
    if py is None:
        py = np.asarray((X.sum(axis=0) / X.sum()).reshape(-1))
    if beta < 1:
        py = py ** beta
        py /= py.sum()
    pxy = X / X.sum()

    # py에 제곱근을 사용
    # SCI (Washtell and Markert, 2009)
    # py = py**(1/2)
    
    
    # transform px and py to diagonal matrix
    # using scipy.sparse.diags
    # pmi_alpha (x,y) = p(x,y) / ( p(x) x (p(y) + alpha) )
    px_diag = _as_diag(px, 0)
    py_diag = _as_diag(py, alpha)
    exp_pmi = px_diag.dot(pxy).dot(py_diag)

    # PPMI using threshold
    min_exp_pmi = 1 if min_pmi == 0 else np.exp(min_pmi)
    pmi = _logarithm_and_ppmi(exp_pmi, min_exp_pmi)

    return pmi, px, py, pxy

# def df2edge_player(data, _dict = False, _map = True, p_mode = 'default'):

#     if _map:
#         # pyear mapping
#         _year = data.loc[:, ['doc_id', 'pyear']].drop_duplicates()
#         _year = _year.set_index('doc_id').pyear.to_dict()

#         # section mapping
#         _section = data.loc[:, ['doc_id', 'section']].drop_duplicates() 
#         _section = _section.set_index('doc_id').section.to_dict()

#     data = data.loc[:, ['doc_id', 'orgn_id']]

#     groups = defaultdict(lambda: [])
#     for doc, orgn in data.values:
#         groups[doc].append(orgn)

#     p_dict = {k:v for k, v in enumerate(data['doc_id'].unique())}
#     o_dict = {k:v for k, v in enumerate(data['orgn_id'].unique())}

#     data['doc_id'] = data['doc_id'].map({v:k for k, v in zip(p_dict.keys(), p_dict.values())})
#     data['orgn_id'] = data['orgn_id'].map({v:k for k, v in zip(o_dict.keys(), o_dict.values())})
    
#     data['value'] = 1
#     data = data.drop_duplicates()

#     s_two_mat = sparse.csr_matrix((data['value'], (data['doc_id'], data['orgn_id'])))
    
#     def make_graph_value(one_mat, _dict, p_mode):
    
#         _n, _g = sparse.csgraph.connected_components(one_mat)
#         one_mat_deg = pd.DataFrame(one_mat.diagonal(),_dict.values())[0].to_dict()
#         one_mat = sparse.triu(one_mat, k = 1, format = 'csr')

#         source, target = one_mat.nonzero()
#     #             source, target = np.nonzero(one_mat>0) 
#         weight = one_mat.data            

#         one_edge = pd.DataFrame({'source': source, 'target': target, 'weight': weight})                
#         one_node = pd.DataFrame({'id': _dict.keys(), 'name': _dict.values(), 'cluster': _g, 'linklen': one_mat_deg.values()})                           
        
#         if p_mode == 'max':
#             one_graph = 1
        
#         elif p_mode == 'default':
#             one_graph = make_one_graph(one_node, one_edge)
#             to_delete_ids = [v.index for v in one_graph.vs if v.degree() == 0]
#             one_graph.delete_vertices(to_delete_ids)
        
#         one_edge.source = one_edge.source.map(_dict)
#         one_edge.target = one_edge.target.map(_dict)
    
#         return one_edge, one_node, one_graph        
    
#     one_mat = np.dot(s_two_mat.T, s_two_mat)
#     o_edge, o_node, o_graph = make_graph_value(one_mat, o_dict, p_mode)
#     # p_node['_neighbor'] = ''

#     if _map:
#         o_node['pyear'] = [list(set(map(_year.get, [n]))) for n in o_node['name']]
#         o_node['section'] = [list(set(map(_section.get, [n]))) for n in o_node['name']]
    
#     if _dict:
#         return o_edge, o_node, o_dict
#     else:
#         return o_edge, o_node, o_graph


def df2edge(data, mode, _dict = False, _map = True, p_mode = 'default'):
    
    if _map:
        # pyear mapping
        _year = data.loc[:,['doc_id','pyear']].drop_duplicates()
        _year = _year.set_index('doc_id').pyear.to_dict()

        # section mapping
        _section = data.loc[:,['doc_id','section']].drop_duplicates() 
        _section = _section.set_index('doc_id').section.to_dict()

    # make graph
    data = data.loc[:,['word','doc_id']]       
    
    groups = defaultdict(lambda: [])
    for word, _id in data.values:
        groups[word].append(_id)

    w_neighbor = pd.DataFrame({'word':groups.keys(), '_neighbor':groups.values()}).set_index('word')
    
    # w_neighbor = data.groupby('word').apply(lambda x: ','.join(x['doc_id'])).rename('_neighbor')
    # w_neighbor = w_neighbor.apply(lambda x:list(set(x.split(','))))
    
    w_dict = {i:v for i,v in enumerate(data.word.unique())}
    p_dict = {i:v for i,v in enumerate(data.doc_id.unique())}
    
    data.word = data.word.map({v:i for i,v in zip(w_dict.keys(),w_dict.values())})

    data.doc_id = data.doc_id.map({v:i for i,v in zip(p_dict.keys(),p_dict.values())})
  
    data['value'] = 1
    data = data.drop_duplicates()

    s_two_mat = sparse.csr_matrix((data.value, (data.word, data.doc_id)))
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


    if mode == 'word':

        one_mat = np.dot(s_two_mat,s_two_mat.T)
        w_edge, w_node, w_graph = make_graph_value(one_mat, w_dict, p_mode = 'default')
        w_node = pd.merge(w_node, w_neighbor, left_on = 'name', right_index = True)
        
        if _map:
            w_node['pyear'] = [sorted(list(set(map(_year.get, n)))) for n in w_node._neighbor]
            w_node['section'] = [sorted(list(set(map(_section.get, n)))) for n in w_node._neighbor]

        return w_edge, w_node, w_graph

    elif mode == 'paper':

        one_mat = np.dot(s_two_mat.T,s_two_mat)
        p_edge, p_node, p_graph = make_graph_value(one_mat, p_dict, p_mode)
        # p_node['_neighbor'] = ''

        if _map:
            p_node['pyear'] = [list(set(map(_year.get, [n]))) for n in p_node.name]
            p_node['section'] = [list(set(map(_section.get, [n]))) for n in p_node.name]
            
        
        if _dict:
            return p_edge, p_node, p_dict
        else:
            return p_edge, p_node, p_graph
    
    elif mode == 'all':
        
        a_mat = [(np.dot(s_two_mat,s_two_mat.T),w_dict), (np.dot(s_two_mat.T,s_two_mat),p_dict)]
        A = [make_graph_value(m,d) for m,d in a_mat]
        
        """
        A[0]은 word_network, A[1]은 paper_network
        각 네트워크에서 0,1,2번째 원소는 순서대로 edge, node, graph정보
        
        A[0][0] : word_network의 edge
        A[0][1] : word_network의 node
        A[0][2] : word_network의 graph
        
        A[1][0] ~ A[1][2]는 paper_network의 결과

        mode_all은 dict 형식으로 저장
        output은 w_edge ~ p_graph까지 있으며, 상황에 따라 호출하면 된다.
        w_edge = mode_all['w_edge']
                
        """        
        w_edge = A[0][0]
        w_node = A[0][1]
        w_graph = A[0][2]
        p_edge = A[1][0]
        p_node = A[1][1]
        p_graph = A[1][2]
        
        w_node = pd.merge(w_node, w_neighbor, left_on = 'name', right_index = True)
        # p_node['_neighbor'] = ""

        w_node['pyear'] = [sorted(list(set(map(_year.get, n)))) for n in w_node._neighbor]
        p_node['pyear'] = [list(set(map(_year.get, [n]))) for n in p_node.name]
        
        mode_all = {'w_edge':w_edge, 'w_node':w_node, 'w_graph':w_graph,
                    'p_edge':p_edge, 'p_node':p_node, 'p_graph':p_graph}
        
        return mode_all

def df2edge2(data, mode, _dict = False, p_mode = 'default'):
    
  
    # make graph
    data = data.loc[:,['word','doc_id']]       
    
    groups = defaultdict(lambda: [])
    for word, _id in data.values:
        groups[word].append(_id)

    w_neighbor = pd.DataFrame({'word':groups.keys(), '_neighbor':groups.values()}).set_index('word')
    
    # w_neighbor = data.groupby('word').apply(lambda x: ','.join(x['doc_id'])).rename('_neighbor')
    # w_neighbor = w_neighbor.apply(lambda x:list(set(x.split(','))))
    
    w_dict = {i:v for i,v in enumerate(data.word.unique())}
    p_dict = {i:v for i,v in enumerate(data.doc_id.unique())}
    
    data.word = data.word.map({v:i for i,v in zip(w_dict.keys(),w_dict.values())})

    data.doc_id = data.doc_id.map({v:i for i,v in zip(p_dict.keys(),p_dict.values())})
  
    data['value'] = 1
    data = data.drop_duplicates()

    s_two_mat = sparse.csr_matrix((data.value, (data.word, data.doc_id)))
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


    if mode == 'word':

        one_mat = np.dot(s_two_mat,s_two_mat.T)
        w_edge, w_node, w_graph = make_graph_value(one_mat, w_dict, p_mode = 'default')
        w_node = pd.merge(w_node, w_neighbor, left_on = 'name', right_index = True)

        return w_edge, w_node, w_graph

    elif mode == 'paper':

        one_mat = np.dot(s_two_mat.T,s_two_mat)
        
        p_edge, p_node, p_graph = make_graph_value(one_mat, p_dict, p_mode)
        # p_node['_neighbor'] = ''

        if _dict:
            return p_edge, p_node, p_dict
        else:
            return p_edge, p_node, p_graph
    
    elif mode == 'all':
        
        a_mat = [(np.dot(s_two_mat,s_two_mat.T),w_dict), (np.dot(s_two_mat.T,s_two_mat),p_dict)]
        A = [make_graph_value(m,d) for m,d in a_mat]
        
        """
        A[0]은 word_network, A[1]은 paper_network
        각 네트워크에서 0,1,2번째 원소는 순서대로 edge, node, graph정보
        
        A[0][0] : word_network의 edge
        A[0][1] : word_network의 node
        A[0][2] : word_network의 graph
        
        A[1][0] ~ A[1][2]는 paper_network의 결과

        mode_all은 dict 형식으로 저장
        output은 w_edge ~ p_graph까지 있으며, 상황에 따라 호출하면 된다.
        w_edge = mode_all['w_edge']
                
        """        
        w_edge = A[0][0]
        w_node = A[0][1]
        w_graph = A[0][2]
        p_edge = A[1][0]
        p_node = A[1][1]
        p_graph = A[1][2]
        
        w_node = pd.merge(w_node, w_neighbor, left_on = 'name', right_index = True)
        # p_node['_neighbor'] = ""

        
        mode_all = {'w_edge':w_edge, 'w_node':w_node, 'w_graph':w_graph,
                    'p_edge':p_edge, 'p_node':p_node, 'p_graph':p_graph}
        
        return mode_all
    
def df2edge_player(data, mode, _dict = False, _map = True, drop_match = True, p_mode = 'default'):
    if drop_match:
        data = data[data['orgn_id'] != data['section']]

    raw_data = data.copy()
    data = data.loc[:, ['orgn_id', 'section']]
    o_dict = {k:v for k, v in enumerate(data['orgn_id'].unique())}
    m_dict = {k:v for k, v in enumerate(data['section'].unique())}

    data['orgn_id'] = data['orgn_id'].map({v:k for k, v in zip(o_dict.keys(), o_dict.values())})
    data['section'] = data['section'].map({v:k for k, v in zip(m_dict.keys(), m_dict.values())})
    data['value'] = 1
    data = data.drop_duplicates()

    s_two_mat = sparse.csr_matrix((data['value'], (data['orgn_id'], data['section'])))
    
    def make_graph_value(one_mat, _dict, p_mode):
    
        _n, _g = sparse.csgraph.connected_components(one_mat)
        one_mat_deg = pd.DataFrame(one_mat.diagonal(), _dict.values())[0].to_dict()
        one_mat = sparse.triu(one_mat, k = 1, format = 'csr')

        source, target = one_mat.nonzero()
    #             source, target = np.nonzero(one_mat>0) 
        weight = one_mat.data            

        one_edge = pd.DataFrame({'source': source, 'target': target, 'weight': weight})                
        one_node = pd.DataFrame({'id': _dict.keys(), 'name': _dict.values(), 'cluster': _g, 'linklen': one_mat_deg.values()})                           
        
        if p_mode == 'max':
            one_graph = 1
        
        elif p_mode == 'default':
            one_graph = make_one_graph(one_node, one_edge)
            to_delete_ids = [v.index for v in one_graph.vs if v.degree() == 0]
            one_graph.delete_vertices(to_delete_ids)
        
        one_edge['source'] = one_edge['source'].map(_dict)
        one_edge['target'] = one_edge['target'].map(_dict)
    
        return one_edge, one_node, one_graph

    if mode == 'orgn':
        one_mat = np.dot(s_two_mat, s_two_mat.T)

        o_edge, o_node, o_graph = make_graph_value(one_mat, o_dict, p_mode = 'default')

        if _map:
            _year = raw_data.loc[:, ['orgn_id', 'pyear']].drop_duplicates()
            _year = _year.groupby(['orgn_id'])['pyear'].agg(list).reset_index().rename({'orgn_id': 'name'}, axis = 1)
            # _year = _year['pyear'].to_dict()
            o_node = o_node.merge(_year, how = 'left')

            _section = raw_data.loc[:, ['orgn_id', 'section']].drop_duplicates()
            _section = _section.groupby(['orgn_id'])['section'].agg(list).reset_index().rename({'orgn_id': 'name'}, axis = 1)
            # _section = _section['section'].to_dict()
            o_node = o_node.merge(_section, how = 'left')

        return o_edge, o_node, o_graph

    elif mode == 'mgnt':    
        one_mat = np.dot(s_two_mat.T, s_two_mat)
        m_edge, m_node, m_graph = make_graph_value(one_mat, m_dict, p_mode)

        if _map:
            _year = raw_data.loc[:, ['section', 'pyear']].drop_duplicates()
            _year = _year.groupby(['section'])['pyear'].agg(list).reset_index().rename({'section': 'name'}, axis = 1)
            m_node = m_node.merge(_year, how = 'left')
            
            # mgnt
            m_node['section'] = None
        
        if _dict:
            return m_edge, m_node, m_dict
            
        else:
            return m_edge, m_node, m_graph

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

def make_one_mode_graph(edgelist):
    # edgelist에서 graph생성 - 확인용
    data = copy.copy(edgelist)
    
    # Node에 대한 DF 생성
    Node_list = sorted(list(set(data.source.tolist()) | set(data.target.tolist())))
    Node_dict = {v:i for i,v in enumerate(Node_list)}

    # Edge 정보 생성 - Weight
    data.source = data.source.map(Node_dict)
    data.target = data.target.map(Node_dict)

    # Edgekey를 Tuple로 변경
    edgelist = data[['source', 'target']]
    EdgeTuple = list(edgelist.itertuples(index = False, name =  None))

    # Graph 생성
    graph = igraph.Graph()
    graph.add_vertices(len(Node_dict))
    graph.add_edges(EdgeTuple)

    # Graph에 Node 정보 입력
    graph.vs["name"] = Node_list

    # Graph에 Edge의 정보 입력
    graph.es["weight"] = data.weight.tolist()

    return graph

def ego_edgelist_node_controll(section_ego_edgelist, graph, filtering_dict, mode='default'):
    temp_df = pd.DataFrame(section_ego_edgelist)   
    
    term_indice_df = temp_df[["word", "dfreq"]].drop_duplicates(['word'])
    term_indice_df = term_indice_df.set_index("word")
    term_indice_df = term_indice_df.replace([np.inf, -np.inf], np.nan).fillna(1)
    
    def convert_fill(df):
        return df.stack().apply(pd.to_numeric, errors = 'ignore').fillna(1).unstack()

    term_indice_df = convert_fill(term_indice_df)
    
    if term_indice_df['dfreq'].sum() >= 300:
        term_indice_df = term_indice_df.loc[(term_indice_df['dfreq'] >= filtering_dict['dfreq'])]
    
    # node의 수를 300개로 잡는 것 기준
    # node의 수가 300개인 경우, 밀도가 0.02를 초과하면 시안성이 좋지 않다고 판단하였고, 0.01미만인 경우는 굳이 pruning을 할만큼 시안성이 나쁘지 않다고 판단하였다.
    # 따라서 0.02초과, 0.02~0.01, 0.01미만으로 나누어 0.02초과인 경우에는 최대 가중치인 0.9, 0.01미만은 가중치가 존재하지 않도록 설정하였다.
    root_parent = 0

    if mode == 'max':
        G_density = 1

    elif mode == 'default':
        G_density = round(graph.density(loops=False), 3)
    
    upper_d = 0.001
    lower_d = 0.0005

    reduce_param = (G_density - lower_d) / (upper_d - lower_d)
    
    if term_indice_df.dfreq.sum() > filtering_dict["root_parent_base"]:
        if G_density >= upper_d:
            root_parent = (((term_indice_df.dfreq.sum() / (filtering_dict["root_parent_base"] - (filtering_dict["root_parent_base"] * 0.9))) * 0.1) - 0.1)
        
        elif G_density < upper_d and G_density > lower_d:
            root_parent = (((term_indice_df.dfreq.sum() / (filtering_dict["root_parent_base"] - (filtering_dict["root_parent_base"] * reduce_param))) * 0.1) - 0.1)
        
        else:
            root_parent = (((term_indice_df.dfreq.sum() / filtering_dict["root_parent_base"]) * 0.1) - 0.1)

        term_indice_df['node_limit'] = term_indice_df['dfreq'] ** (1 / (1 + root_parent))
    
    else:
        term_indice_df['node_limit'] = term_indice_df['dfreq'] ** (1 / filtering_dict["node_limit_base"])  
    
    term_indice_df['node_limit'] = term_indice_df['node_limit'].apply(np.ceil)
    term_indice_df['node_limit'] = term_indice_df['node_limit'].apply(lambda x: int(x))
    
    return term_indice_df

def complete_egoedgelist(section_ego_edgelist, term_indiced_df, filtering_dict):
    target_ego_edgelist = section_ego_edgelist[section_ego_edgelist.word.isin(term_indiced_df.index)]
    target_ego_edgelist = target_ego_edgelist.set_index("word")
    target_ego_edgelist["node_limit"] = term_indiced_df["node_limit"]
    target_ego_edgelist = target_ego_edgelist.reset_index()
    target_ego_edgelist = target_ego_edgelist.sort_values(['word', 'bm25'], ascending = False)
    target_ego_edgelist = target_ego_edgelist.reset_index(drop = True)

    duplicate_list = target_ego_edgelist.duplicated('word', keep = 'first')
    duplicate_list = duplicate_list.index[~duplicate_list].tolist()
    duplicate_list_limit = target_ego_edgelist.iloc[duplicate_list]['node_limit']
    duplicate_list_limit = list(np.array(duplicate_list) + np.array(duplicate_list_limit))
    
    target_list = [list(range(duplicate_list[n], duplicate_list_limit[n])) for n in range(0, len(duplicate_list))]
    target_list = list(itertools.chain(*target_list))

    final_ego_edgelist = target_ego_edgelist.iloc[target_list]
    final_ego_edgelist=final_ego_edgelist.drop(['node_limit'],axis = 1)
    final_ego_edgelist = final_ego_edgelist.reset_index(drop = True)
    
    return final_ego_edgelist

def bm25(N, dtfreq, dfreq, k = 1.2):

    # 모든 문서가 평균적인 문서길이를 가진다고 가정
    # 데이터가 추가되면 인수 추가
    # 나중에는 elasticsearch에서 계산된 값을 사용

    _idf = np.log(((N - dfreq + 0.5) / (dfreq + 0.5)) + 1)
    _score = _idf * ((dtfreq * (k + 1)) / (dtfreq + k))
    
    return _score