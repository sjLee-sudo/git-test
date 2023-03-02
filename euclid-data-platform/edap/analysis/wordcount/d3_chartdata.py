import pandas as pd
import numpy as np
import json
import itertools
from pandas.core.algorithms import isin

from tqdm import tqdm

if __name__ !='__main__':
    from db_manager.managers import ElasticQuery
    from terms.models import EdgelistTerm
    from config.constants import WORDCOUNT_EDGELIST_COLUMN_MAPPING, SNA_EDGELIST_COLUMN_MAPPING, CLOUD_CHART_WORD_CNT_LIMIT
    from analysis.edgelist.kistep_edgelist import WordCountEdgelist
    from documents.document_utils import create_keyword_search_query
    from config.constants import MAX_CLAUSE_COUNT

"""
# line chart
# line chart + bar chart가 혼합된 모습
# line chart는 그대로 해당 단어가 등장하는 과제 수이고, bar chart는 해당 단어가 등장하는 과제들의 예산 합이다.
# 두 chart 사이의 단위가 다르기 때문에 비교가 가능하도록 단위를 적절히 맞추어줄 필요성이 있다.
# 여러 단어를 선택했을 때, line chart는 여러 개 그려지지만 bar chart는 여러 단어들이 등장하는 과제들의 예산 합이다.
# 따라서 예산이 중복해서 더해지지 않도록 신경써야 한다.
# 해당 연도에 등장한 모든 과제 id와 모든 과제 예산 정보를 list로 넘겨줘서 unique한 애들만 다시 계산해서 넘겨주는 식으로 시각화 단계에서 처리

* element

- date : 연도 | str
- sort : section 정보 | str
- value : 문서 수 | int

* example
[{'date': '2006', 'sort': 'U', 'value': 1},
 {'date': '2010', 'sort': 'U', 'value': 1},
 {'date': '2015', 'sort': 'U', 'value': 2},
 {'date': '2016', 'sort': 'N', 'value': 3},
 {'date': '2017', 'sort': 'N', 'value': 4},
 {'date': '2018', 'sort': 'N', 'value': 3}]

# line chart ver.2

* element
> line
- x : 연도 | list(str)
- y : 연도별 과제 수 | list(int)
- ys : 연도별 과제 수 표준화 | list(float)
- doc : 해당 연도에 등장한 과제 id | 2D-list
- type : chart type | str

> bar
- x : 연도 | list(str)
- y : 연도별 과제비 총액(단위 : 천만원) | list(float)
- ys : 연도별 과제비 총액 표준화 | list(float)
- doc : 해당 연도에 등장한 과제 id | 2D-list
- type : chart type | str

* example
{'3d': {'bar': {'x': ['2018', '2019', '2020'],
   'y': [21.0, 11.0, 37.0],
   'ys': [0.0006420448, 0.0003363092, 0.0011312217],
   'doc': [['1711073322', '1425121376'],
    ['1425129442'],
    ['1415169693', '1465032252']],
   'type': 'bar'},
  'line': {'x': ['2018', '2019', '2020'],
   'y': [2, 1, 2],
   'ys': [0.0008510638, 0.0, 0.0008510638],
   'doc': [['1711073322', '1425121376'],
    ['1425129442'],
    ['1415169693', '1465032252']],
   'type': 'scatter'}},
 'aas': {'bar': {'x': ['2020'],
   'y': [13.0],
   'ys': [0.0003974563],
   'doc': [['1415168423']],
   'type': 'bar'},
  'line': {'x': ['2020'],
   'y': [1],
   'ys': [0.0],
   'doc': [['1415168423']],
   'type': 'scatter'}}}

# bubble chart
* input
- wc_edge_df | data.frame

* output element

> 각 용어가 등장하는 문서의 연도별 발생 추이
> '인공지능'이라는 단어가 전년도 대비 얼마나 증가했고, 전체 기간동안 얼마나 등장했는지 등의 정보
> 상위 20개 단어들만 반환

- id : word | str
- year : 연도 | str
- size : bubble의 크기(과제 수) | int
- x : x좌표(과제 수 누적합) | int
- y : y좌표(증가율) | float
- t_size : 초기 선택 여부를 결정하기 위한 값, 전체 단어 등장 수 | int

* example

[{'id': 'loa',
  'year': '2004',
  'size': 2,
  'group': 'loa',
  'x': 2,
  'y': 0.0,
  't_size': 4,
  'viz': ''},
 {'id': 'loa',
  'year': '2005',
  'size': 1,
  'group': 'loa',
  'x': 3,
  'y': -50.0,
  't_size': 4,
  'viz': ''},
 {'id': 'loa',
  'year': '2006',
  'size': 1,
  'group': 'loa',
  'x': 4,
  'y': 0.0,
  't_size': 4,
  'viz': ''}]

# bar chart

* input
- wc_edge_df | data.frame
- n : 연도별 최대 class 등장 수 | int

* output element
> class + 연도별 수행 과제 수
> 각 연도 내 class 별로 누적막대그래프가 그려진다.
> 각 연도별로 빈도가 가장 높은 5개의 class만 그려진다.

- name : section | str
- x : 연도 | list(str)
- y : (연도별) 과제 수 | list(int)
- type : plotly 타입 지정 | str

* example
[{'name': 'E',
  'x': ['2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020'],
  'y': [11, 6, 7, 8, 5, 8, 6, 7, 32, 29, 152, 252, 404],
  'type': 'bar'},
 {'name': 'H', 'x': ['2011', '2013', '2014'], 'y': [1, 1, 2], 'type': 'bar'}

"""

def convert_json(d):
    _json = d.to_json(orient='records')
    _json = json.loads(_json)
    return _json


def line_chart(wc_edge_df, mode='word'):
    
    _data = wc_edge_df.copy()    

    if mode == 'word':
        _data.rndco_tot_amt = _data.rndco_tot_amt / 1e+07 # 단위 : 천만원
        _data = _data.drop_duplicates(['doc_id','term'])

        line_data = _data.loc[:,['term','publication_date']].groupby(['term','publication_date']).size().rename('dfreq').reset_index()
        budget = _data.loc[:,['term','publication_date','rndco_tot_amt']].groupby(['term','publication_date']).sum('rndco_tot_amt').rndco_tot_amt.tolist()
        line_data['budget'] = np.round(budget,0)
        
        line_data = line_data.rename(columns={'dfreq':'value','publication_date':'date'})
        _line = line_data.groupby(['term','date']).sum('value').reset_index().set_index('term')

        g_line = _line.groupby('term')
        t_line = _line.index.unique()    
        
        _val = g_line.apply(lambda x:convert_json(x))

        line_json = {t:v for t,v in zip(t_line,_val)}
    
    elif mode == 'all':
        _data = _data.rename(columns = {'publication_date':'date'})
        _data = _data.drop_duplicates(['doc_id','date']).loc[:,['doc_id','date']]
        _data = _data.groupby('date').size().rename('value').reset_index()

        line_json = convert_json(_data)

    return line_json

def line_chart_ver2(_data):
    
    _data.rndco_tot_amt = _data.rndco_tot_amt / 1e+07 # 단위 : 천만원
    _data = _data.drop_duplicates(['doc_id','term'])

    v_doc = iter(_data.sort_values(['term','publication_date']).doc_id.tolist())
    n_doc = _data.groupby(['term','publication_date']).size().tolist()
    v_doc = [list(itertools.islice(v_doc,i)) for i in n_doc]

    line_data = _data.loc[:,['term','publication_date']].groupby(['term','publication_date']).size().rename('dfreq').reset_index()

    budget = _data.loc[:,['term','publication_date','rndco_tot_amt']].groupby(['term','publication_date']).sum('rndco_tot_amt').rndco_tot_amt.tolist()
    line_data['budget'] = np.round(budget,0)

    line_data = line_data.rename(columns={'dfreq':'value','publication_date':'date'})

    _line = line_data.groupby(['term','date']).sum('value').reset_index().set_index('term')
    _line['doc'] = v_doc

    def minmax_value(value):
        with np.errstate(divide='ignore', invalid='ignore'): 
            minmax = (value-value.min())/(value.max()-value.min())
            np.nan_to_num(minmax, 0)
        return minmax.tolist()

    n_line = _line.groupby('term').size()
    t_line = _line.index.unique()

    # ############ 전체 표준화
    c_s_line = iter(minmax_value(_line.value))
    c_s_line = [list(itertools.islice(c_s_line,i)) for i in n_line]
    b_s_line = iter(minmax_value(_line.budget))
    b_s_line = [list(itertools.islice(b_s_line,i)) for i in n_line]
    # ############

    ############ 각각 표준화
    # c_line = iter(_line.value.tolist())
    # y_line = iter(_line.date.tolist())
    # b_line = iter(_line.budget.tolist())

    # c_s_line = [np.array(list(itertools.islice(c_line,i))) for i in n_line]
    # c_s_line_max = [str(c.max)*n for n,c in zip(n_line,c_s_line)]


    # b_s_line = [np.array(list(itertools.islice(b_line,i))) for i in n_line]
    # 하나씩 돌아가면서 돌리는게 아니라 계산 필요한걸 리스트로 다 생성해서 np.array끼리 계산시키기
    # c_s_line = list(map(minmax_value, c_s_line))
    # b_s_line = list(map(minmax_value, b_s_line))
    ############

    def convert_2d_list(list_1d, list_n):
        list_1d = iter(list_1d)
        return [list(itertools.islice(list_1d,i)) for i in list_n]

    c_line = iter(_line.value.tolist())
    y_line = iter(_line.date.tolist())
    b_line = iter(_line.budget.tolist())
    d_line = iter(_line.doc.tolist())

    c_line = [list(itertools.islice(c_line,i)) for i in n_line]
    y_line = [list(itertools.islice(y_line,i)) for i in n_line]
    b_line = [list(itertools.islice(b_line,i)) for i in n_line]
    d_line = [list(itertools.islice(d_line,i)) for i in n_line]

    line_part = pd.DataFrame([(y,c,cs,d,'scatter') for y,c,cs,d in zip(y_line,c_line,c_s_line,d_line)], columns = ['x','y','ys','doc','type'])
    bar_part = pd.DataFrame([(y,b,bs,d,'bar') for y,b,bs,d in zip(y_line,b_line,b_s_line,d_line)], columns = ['x','y','ys','doc','type'])

    line_part = convert_json(line_part)
    bar_part = convert_json(bar_part)

    all_json = [{'bar':bar,'line':line} for bar,line in zip(bar_part,line_part)]
    all_json = {t:v for t,v in zip(t_line,all_json)}

    return all_json

def line_chart_ver3(wc_edge_df):
    line_chart = {'word_line_chart':{},'paper_line_chart':{},'ipr_line_chart':{},'sbjt_line_chart':{}}
    if wc_edge_df.empty:
        return line_chart
    line_df = wc_edge_df[['doc_id','term','publication_date']]
    line_df = line_df.rename(columns={'publication_date':'date'})
    term_group_df = line_df.groupby(['term','date'])['doc_id'].count().reset_index().rename(columns={'doc_id':'y'})
    line_chart_df = term_group_df.groupby('term').agg(list)
    word_chart = line_chart_df.to_dict('index')
    # data by year
    yearly_df = wc_edge_df.drop_duplicates(['doc_id'])
    yearly_ipr_paper_df = yearly_df.groupby(['publication_date'])[['ipr','paper']].sum()
    yearly_sbjt_series = yearly_df.groupby('publication_date')['doc_id'].size()
    yearly_ipr_paper_df['sbjt'] = yearly_sbjt_series
    yearly_combined_df = yearly_ipr_paper_df.reset_index().rename({'publication_date':'date'},axis=1)
    ipr_chart = yearly_combined_df[['date','ipr']].rename({'ipr':'yCount'},axis=1).to_dict('records')
    paper_chart = yearly_combined_df[['date','paper']].rename({'paper':'yCount'},axis=1).to_dict('records')
    sbjt_chart = yearly_combined_df[['date','sbjt']].rename({'sbjt':'yCount'},axis=1).to_dict('records')
    # chart_source_df = chart_source_df.drop_duplicates(['doc_id'])
    # chart_source_df = chart_source_df.set_index('doc_id')
    # chart_source_dict = chart_source_df.to_dict('index')
    line_chart.update({'word_line_chart': word_chart,'paper_line_chart':paper_chart,'ipr_line_chart':ipr_chart,'sbjt_line_chart':sbjt_chart})
    return line_chart

def bubble_chart(viz_df):
    
    word_doc = viz_df.copy().loc[:,['doc_id','term','subclass','publication_date']]
    word_doc['sort'] = [d[0] for d in word_doc.subclass]

    _bubble = word_doc.groupby(['term','publication_date']).size().rename('value').reset_index()
    _bubble['group'] = _bubble.term

    v_bubble = iter(_bubble.value.tolist())
    n_bubble = _bubble.groupby('group').size().tolist()

    _val = [np.array(list(itertools.islice(v_bubble,i))) for i in n_bubble]

    _x = [v.cumsum().tolist() for v in _val]
    _x = list(itertools.chain(*_x))

    def _shift(x,n):
        _a = np.roll(x,n)
        _a[n-1] = 0
        return _a

    with np.errstate(divide='ignore'):         
        _y = [np.nan_to_num(((v/_shift(v,1))-1)*100, posinf=0, neginf=0).tolist() for v in _val]
    _y = list(itertools.chain(*_y))

    _bubble['x'] = _x
    _bubble['y'] = _y
    _bubble['y'] = _bubble['y'].map(lambda x: round(x,2))
    t_size = word_doc.groupby('term').size().rename('t_size')

    _bubble = _bubble.set_index('term')
    _bubble = pd.merge(_bubble, t_size, right_index=True, left_index=True).reset_index()
    _bubble = _bubble.rename(columns = {'publication_date':'year','sort':'class','value':'size','term':'id'})

    # 상위 n개만 우선 표출?
    _defalut = t_size.sort_values(ascending=False).head(20).index.tolist()
    # _bubble['viz'] = ['T' if t in _defalut else '' for t in _bubble.id]
    _bubble = _bubble[_bubble.id.isin(_defalut)]
    id_list = _bubble.id.unique()
    year_list = _bubble.year.unique()
    
    year_bubble_list = []
    for year in year_list:
        year_bubble = pd.DataFrame({'id':id_list})
        year_bubble['year'] = year
        year_bubble['group'] = year_bubble['id']
        year_bubble_list.append(year_bubble)
    merged_df = pd.concat(year_bubble_list,axis=0)
    merged_df = merged_df.set_index(['id','year'])

    _indexed_bubble = _bubble.set_index(['id','year'])
    merged_df[['x','y','t_size','size']] = _indexed_bubble[['x','y','t_size','size']]
    merged_df = merged_df.reset_index().sort_values(['id','year'])
    merged_df = merged_df.fillna(0)
    x_min = merged_df['x'].min()
    x_max = merged_df['x'].max()
    y_min = merged_df['y'].min()
    y_max = merged_df['y'].max()
    # make json
    bubble_json = {'chart_data':{},'range_data':{'x_min':x_min, 'x_max':x_max, 'y_min':y_min, 'y_max':y_max}}
    chart_data = merged_df.to_dict(orient='records')
    bubble_json.update({'chart_data':chart_data})
    return bubble_json


def bar_chart(viz_df, category_map={}, n=10):
    
    bar_data = viz_df.copy()
    bar_data = bar_data.drop_duplicates(['doc_id','publication_date','subclass']).loc[:,['doc_id','publication_date','subclass']]
    bar_data = bar_data.rename(columns = {'publication_date':'year'})
    bar_data['sort'] = [s[:2] for s in bar_data.subclass]

    def get_max_n(data, n):
        if len(data) > n:
            data_n = data.iloc[0:n]
        else:
            data_n = data
        return data_n
        
    score_bar = bar_data.drop_duplicates(['doc_id','year','sort'])
    score_bar = score_bar.groupby(['sort','year']).size().rename('value').reset_index()
    score_bar = score_bar.sort_values(['year','value'],ascending=[True, False])
    score_bar = score_bar.groupby('year').apply(lambda x:get_max_n(x,n)).reset_index(drop=True)

    bar_x = score_bar.groupby('sort').apply(lambda x:x.year.tolist()).rename('x')
    bar_y = score_bar.groupby('sort').apply(lambda x:x.value.tolist()).rename('y')

    _bar = pd.merge(bar_x,bar_y,left_index=True, right_index=True).reset_index()
    _bar = _bar.rename(columns = {'sort':'name'})
    _bar['type'] = 'bar'
    if len(category_map)>0:
        _bar['name'] = _bar['name'].map(category_map)
        _bar['name'] = _bar['name'].fillna('분류체계없음')
    bar_json = convert_json(_bar)
    return bar_json


def make_cloud_chart(wc_edge_df):
    wc_edge_df = wc_edge_df.drop_duplicates()
    cloud_chart_list = []
    if wc_edge_df.empty:
        return cloud_chart_list
    cloud_doc_id_df = wc_edge_df[['doc_id','term']]
    doc_freq_sum = cloud_doc_id_df.groupby("term").doc_id.count()
    cloud_doc_id_df = cloud_doc_id_df.set_index("term")
    cloud_doc_id_df["doc_freq_sum"] = doc_freq_sum
    _max = 12
    _wordsum_max = cloud_doc_id_df.doc_freq_sum.max()
    _denom = _wordsum_max / _max
    cloud_doc_id_df = cloud_doc_id_df.sort_values("doc_freq_sum",ascending=False)
    cloud_doc_id_df["norm_doc_freq_sum"] = (np.ceil(cloud_doc_id_df.doc_freq_sum / _denom)).astype(int)
    cloud_doc_id_df['doc_ids'] = wc_edge_df.groupby('term')['doc_id'].agg(list)
    cloud_doc_id_df = cloud_doc_id_df.reset_index()
    cloud_doc_id_df = cloud_doc_id_df[['term','doc_freq_sum','norm_doc_freq_sum','doc_ids']]
    cloud_doc_id_df = cloud_doc_id_df.drop_duplicates('term')
    cloud_chart_list = cloud_doc_id_df.to_dict('records')[:CLOUD_CHART_WORD_CNT_LIMIT]
    return cloud_chart_list


def chart_source_map(wc_edge_df):
    if wc_edge_df.empty:
        return {}
    chart_source_df = wc_edge_df[['doc_id','publication_date','rndco_tot_amt','ipr','paper']]
    chart_source_df = chart_source_df.rename(columns={'publication_date':'date'})
    chart_source_df = chart_source_df.drop_duplicates(['doc_id'])
    chart_source_df = chart_source_df.set_index('doc_id')
    chart_source_dict = chart_source_df.to_dict('index')
    return chart_source_dict

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
    from db_manager.managers import ElasticQuery
    from analysis.edgelist.kistep_edgelist import WordCountEdgelist

    # 이 아래에 필요한 라이브러리 추가
    import timeit
    from terms.models import EdgelistTerm
    from config.constants import DOCUMENT_DB_NAMES, MAX_CLAUSE_COUNT, CLOUD_CHART_WORD_CNT_LIMIT
    from documents.document_utils import create_keyword_search_query
    start_time = timeit.default_timer()
    # docu_patent, docu_thesis, docu_book, docu_subject
    target_db = 'kistep_sbjt'
    es_query = ElasticQuery(target_db)
    # target_category 
    # result = es_query.calculate_yearly_category_doc_freq(word_list, target_category=target_category)
    # df = pd.DataFrame(result)
    wc = WordCountEdgelist()
    search_text = '인공지능'
    search_form = {
        "search_text" : [search_text]
    }

    # query = create_keyword_search_query(search_form=search_form, size=10000)
    # search_query = query['query']
    # min_score = es_query.get_percentile_score(query=search_query,filter_percent=10)
    # scroll_result = es_query.scroll_docs_by_query(query=search_query,source=['doc_id','stan_yr','doc_subclass'], min_score=min_score)
    # docs_df = pd.DataFrame(scroll_result)

    # search_result = es_query.get_docs_by_full_query(query=query, doc_size=doc_size)
    search_result = es_query.get_docs_by_search_text(search_text=search_text, size=3000, fields=["analysis_target_text"],source=["*"])
    doc_ids = [x['_id'] for x in search_result]

    wc_edge_df = wc.get_wc_refined_ego_edgelist( target_db=target_db, doc_ids=doc_ids, size=len(doc_ids),offsets=False, positions=False)   

    # wc_edge_df.to_parquet('../sna/viz_df.parquet')
    # print(term_doc_id_mapping['bel'])
    # pd.DataFrame(term_doc_id_mapping).T.reset_index().rename(columns={'index':'term'}).to_parquet('../sna/word_doc.parquet')
    _load = timeit.default_timer()-start_time
    
    # wordcloud_result = make_cloud_chart(wc_edge_df,  term_doc_id_mapping)    
    # line_json = line_chart_ver2(wc_edge_df)    
    bubble_json = bubble_chart(wc_edge_df)
    # bar_json = bar_chart(wc_edge_df, 5)
    # chart_source_map(wc_edge_df)

    # line_data = line_chart_ver3(wc_edge_df=wc_edge_df)
    # print(line_data)
    _end = timeit.default_timer()-start_time
    
    print(np.round(_load,2), np.round(_end,2), np.round(_end-_load,2))


    # print(term_doc_id_mapping)
    # print(wordcloud_result)
    print(timeit.default_timer()-start_time)