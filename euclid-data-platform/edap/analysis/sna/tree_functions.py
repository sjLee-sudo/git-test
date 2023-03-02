import pandas as pd
import copy
from scipy import sparse
from itertools import product
import json

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



def node_maker(_data, w_dict, p_dict, target_db):

    if _data.empty:
        emp = pd.DataFrame({'id':target_db,'name':target_db,'link_size':0,'index_key':target_db,'depth':'1-2'}, index=[0])
        return emp
    
    s_link = _data.loc[:,['section','doc_id']].drop_duplicates()
    s_link = s_link.groupby('section').size().rename('link_size')
    S = s_link.reset_index().rename(columns={'section':'name'})
    S['id'] = [target_db + 'S' + s for s in S.name]
    S['index_key'] = 'section'
    S['depth'] = '2-3'
    S = S.loc[:,['id','name','link_size','index_key','depth']]

    W = pd.DataFrame.from_dict(w_dict, orient='index').reset_index().rename(columns={'index':'id',0:'name'})

    P = pd.DataFrame.from_dict(p_dict, orient='index').reset_index().rename(columns={'index':'id',0:'doc_id'})
    P = pd.merge(P, _data.loc[:,['doc_id','section','pyear']].drop_duplicates())

    W.id = [target_db + 'W' + str(i) for i in W.id]
    P.id = [target_db + 'P' + str(i) for i,d,s,p in P.values]
    P = P.drop('section', axis=1)

    W['index_key'] = 'word'
    W['link_size'] = 1
    W['depth'] = '4-5'

    p_link = _data.groupby('doc_id').size().rename('link_size')
    P = pd.merge(P, p_link, left_on = 'doc_id', right_index=True)
    P = pd.merge(P,_data.loc[:,['doc_id','title']].drop_duplicates(),on='doc_id', how='inner')
    P = P.rename(columns = {'doc_id':'index_key', 'title':'name'})
    P['depth'] = '3-4'

    head_node = pd.DataFrame({'id':target_db,'name':target_db,'link_size':0,'index_key':target_db,'depth':'1-2'}, index=[0])
    _node = pd.concat([head_node, S,P,W])   
    
    return _node

def edge_word_paper(_data, _section, w_dict, p_dict, target_db):

    data = copy.copy(_data)

    if _section != 'T':
        data = data[data.section == _section]
        data = data.loc[:,['word','doc_id']]       

    elif _section == 'T':        
        data = data.loc[:,['word','doc_id']]

    if data.empty:
        # print(f'section {_section} is empty')
        pass

    else:
        data.word = data.word.map({v:i for i,v in zip(w_dict.keys(),w_dict.values())})
        data.doc_id = data.doc_id.map({v:i for i,v in zip(p_dict.keys(),p_dict.values())})
        data['value'] = 1

        # row : word, col : document
        s_two_mat = sparse.csr_matrix((data.value, (data.word, data.doc_id)))

        _edge = s_two_mat.nonzero()
        _edgelist = pd.DataFrame({'source':_edge[1], 'target':_edge[0]})
        _edgelist.source = [target_db + 'P' + str(i) for i in _edgelist.source]
        _edgelist.target = [target_db + 'W' + str(i) for i in _edgelist.target]
        _edgelist['weight'] = 1
        _edgelist['rel'] = '4-paper-word'

        if _section == 'T':
            _edgelist['section'] = ''
        else:
            _edgelist['section'] = _section        

        return _edgelist

def edge_maker(_data, w_dict, p_dict, target_db):

    if _data.empty:
        return pd.DataFrame(columns = ['source','target','rel','weight'])
        
    SE = copy.copy(_data)

    SE.doc_id = SE.doc_id.map({v:i for i,v in zip(p_dict.keys(),p_dict.values())})
    SE = SE.loc[:,['section','doc_id']]
    SE = SE.rename(columns = {'section':'source', 'doc_id':'target'})
    SE = SE.drop_duplicates()

    SE.source = [target_db + 'S' + s for s in SE.source]
    SE.target = [target_db + 'P' + str(i) for i in SE.target]

    SE['weight'] = 1
    SE['rel'] = '3-section-paper'
    SE['section'] = ''

    _section = _data.section.unique().tolist()

    section_list = ['T'] + _section
    E = [edge_word_paper(_data, s, w_dict, p_dict, target_db) for s in section_list]

    items = [[target_db], [target_db + 'S' + s for s in _section]]
    edge_head = pd.DataFrame.from_records(list(product(*items)), columns = ['source','target'])
    edge_head['rel'] = '2-index-section'
    edge_head['weight'] = 1
    edge_head['section'] = ''

    _edge = [edge_head, SE] + E
    _edge = pd.concat(_edge)

    return _edge 

def outlink_maker(paper_json, p_dict, target_db):

    if not paper_json:
        return pd.DataFrame(columns=['source','target','v'])

    outer_links = pd.DataFrame(paper_json['links'])

    outer_links.source = outer_links.source.map({v:i for i,v in zip(p_dict.keys(),p_dict.values())})
    outer_links.target = outer_links.target.map({v:i for i,v in zip(p_dict.keys(),p_dict.values())})

    outer_links.source = [target_db +'P'+str(i) for i in outer_links.source]
    outer_links.target = [target_db +'P'+str(i) for i in outer_links.target]

    return outer_links
    