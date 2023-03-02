import copy
import networkx as nx
import pandas as pd
if __name__ != '__main__':
    ...

def unnesting(df, explode, axis):
    if axis==1:
        df1 = pd.concat([df[x].explode() for x in explode], axis=1)
        combined_df = df1.join(df.drop(explode, 1), how='left')
        combined_df = combined_df.drop_duplicates()
        return combined_df
    else :
        df1 = pd.concat([
                         pd.DataFrame(df[x].tolist(), index=df.index).add_prefix(x) for x in explode], axis=1)
        combined_df = df1.join(df.drop(explode, 1), how='left')
        combined_df = combined_df.drop_duplicates()
        return combined_df

def create_compounds_from_graph(node_list, edge_list):
    gp = nx.DiGraph()
    gp.add_nodes_from(node_list)
    gp.add_edges_from(edge_list)
    _last_children = [n for n,d in gp.out_degree() if d ==0]
    no_parents = [n for n,d in gp.in_degree() if d ==0]
    last_children = list(set(_last_children)-set(no_parents))
    ancestors = {}
    compounds = []
    for n in last_children:
        ancestor = nx.ancestors(gp,n)
        if len(ancestor) == 0 or len(ancestor)>4: 
            continue
        ancestor.add(n)
        ancestors.update({n:sorted(ancestor)})
    for k,v in ancestors.items():
        word_list = []
        for _n in v:
            word_list.append(gp.nodes[_n]['word'])
            compound = " ".join(word_list)
            # 마지막 조합 단어만 선택
            if compound.find(" ") == -1 or len(compound.split(" "))!=len(v):
                continue
            compounds.append({'doc_id':gp.nodes[_n]['doc_id'],'word':compound,'col_type':'compound','dtfreq': 0, 'position':[-1]})
    return compounds

class EdgelistCompoundExtractor():
    """
    ego_edgelist 형태에서 단어의 position 정보를 활용해 복합어를 생성하는 클래스

    """
    def __init__(self, refined_ego_edgelist) -> None:
        self.node_list = []
        self.edge_list = []
        self._extract_graph_source(refined_ego_edgelist)
        self.compounds = create_compounds_from_graph(self.node_list, self.edge_list)
        

    def _extract_graph_source(self, refined_ego_edgelist):
        if refined_ego_edgelist.empty:
            return None
        target_df = copy.deepcopy(refined_ego_edgelist[['doc_id','word','position']])
        target_df = unnesting(target_df, ['position'],1)
        target_df = target_df.sort_values(['doc_id','position'],ascending=True).reset_index(drop=True)
        
        target_df = target_df.reset_index().rename(columns={'index':'node_id'})
        target_df['next_position'] = target_df.groupby(['doc_id'])['position'].shift(-1)
        target_df['next_node_id'] = target_df.groupby(['doc_id'])['node_id'].shift(-1)
        target_df[['next_position','next_node_id']] = target_df[['next_position','next_node_id']].fillna(-1).astype(int)
        def check_edge(node_id,next_node_id,position,next_position):
            if next_position != -1 and next_position is not None and position >= next_position -1:
                self.edge_list.append((node_id,next_node_id))
        target_df.apply(lambda x: self.node_list.append((
            x['node_id'],{'doc_id':x['doc_id'],'word':x['word']}
            )),axis=1)
        target_df.apply(lambda x: check_edge(x['node_id'],x['next_node_id'],x['position'],x['next_position']),axis=1)

    def get_compounds(self, limit=100):
        df = pd.DataFrame({'word': [x['word'] for x in self.compounds],'freq':1})
        if df.empty:
            return []
        return df.groupby('word')['freq'].sum().reset_index().sort_values('freq',ascending=False)['word'].tolist()[:limit]
    
    def get_compounds_df(self, limit=100):
        df = pd.DataFrame(self.compounds)
        # 2개 이상의 문서에서 등장한 복합어만 대상
        target_comp_series = df.drop_duplicates(['doc_id','word']).groupby('word')['doc_id'].count()>1
        df = df[df.word.isin(target_comp_series[target_comp_series==True].index)]
        df = df.drop_duplicates(['doc_id','word'])
        return df

if __name__ == '__main__':
    import sys
    import os
    from pathlib import Path
    APP_DIR = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(APP_DIR))
    os.environ['DJANGO_SETTINGS_MODULE'] = 'config.settings.base'
    os.environ['DJANGO_ALLOW_ASYNC_UNSAFE'] = 'true'
    import django
    django.setup()
    import timeit
    

    start_time = timeit.default_timer()
    target_db = 'docu_subject'
    # search_text = '인공지능'
    # query_string = "((검색* OR 경계블록* OR 계산* OR 단말* OR 데이터* OR 도어* OR 연결* OR 이용자* OR 인터넷* OR 자의* OR 지하* OR 차량* OR 충진* OR 콘크리트*) OR ((검색 OR 검색) AND (경계블록 OR 콘크리트) AND (계산 OR 계산) AND (단말 OR 연결 OR 도어 OR 지하) AND (데이터 OR 단말) AND (도어 OR 도어) AND (연결 OR 연결) AND (이용자 OR 이용자) AND (인터넷 OR 인터넷) AND (자의 OR 인터넷 OR 자의 OR 생년월일) AND (지하 OR 지하) AND (차량 OR 차량) AND (충진 OR 연결) AND (콘크리트 OR 콘크리트)))"

    # sna_edge = SnaEdgelist()
    # ego_edgelist_df, refined_ego_edgelist = sna_edge.get_sna_refined_ego_edgelist(target_db=target_db, search_text=search_text)
    refined_ego_edgelist = pd.read_parquet('/data/KISTEP_TEMP/parquet_files/sna_ref_df.parquet')
    start_time = timeit.default_timer()
    compound_extractor = EdgelistCompoundExtractor(refined_ego_edgelist)
    print(compound_extractor.get_compounds())
    print(timeit.default_timer()-start_time)