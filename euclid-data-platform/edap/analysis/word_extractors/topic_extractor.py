# 여기에 토픽, 연결어, 불용어 추출 모듈 로직 작성

import pandas as pd
import numpy as np
import itertools

import itertools
from collections import Counter
from soynlp.noun import LRNounExtractor_v2


if __name__ != '__main__':
    from analysis.sna.sna_functions import make_pmi
    from terms.userdict_utils import delete_word_after_making_userdict
"""
Input data는 Edgelist, EgoEdgelist, ResizeEdgelist로 구분
* Edgelist : 전체 문서에 대한 엣지리스트
- Edgelist_class는 Edgelist에서 class단위로 BM25를 계산한 dataframe
* EgoEdgelist : 검색된 문서에 대한 엣지리스트
* ResizeEdgelist : 검색된 문서에서 주제어만 남긴 엣지리스트

# Edgelist columns
word        type : string
doc_id      type : string
dtfreq      type : int
dfreq       type : int
position    type : list
section     type : string
bm25        type : float
d_class     type : string
col_types   type : string

word    doc_id  dtfreq  dfreq   position    section     bm25    d_class     col_types
수소    100203  5       10      [1,3,4]     A           3.231   A01B        content
자동차  100203  3       12      [1,4,7]     A           6.321   A01B        title

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

# ResizeEdgelist columns는 EgoEdgelist와 동일

"""
def make_rawdata_class(Edgelist):
    
    # 1. ctfreq 생성
    CT = Edgelist.groupby(['d_class','word']).sum()['dtfreq'].rename('ctfreq')
    CT = CT.reset_index()
    Edgelist = pd.merge(Edgelist, CT, on = ['word','d_class'])
    
    # class 별 ttfreq filter 추가, 클래스 내에서 단 한 번만 등장하는 단어들을 제거    

    Edgelist_class = Edgelist.loc[:,['word','ctfreq','d_class','section']]
    Edgelist_class = Edgelist_class.drop_duplicates() # class별로 고유한 단어만 남는다

    # 2. cfreq 생성
    cfreq = Edgelist_class.groupby('word').size().rename('cfreq').to_frame()

    Edgelist_class = Edgelist_class.set_index('word')
    Edgelist_class = pd.merge(Edgelist_class, cfreq, right_index = True, left_index = True)
    Edgelist_class = Edgelist_class.reset_index()
    
    def bm25(N,dtfreq,dfreq,k=1.2):

        # 각 클래스별로 클래스 내 문서들이 평균적인 문서길이를 가진다고 가정

        _idf = np.log(((N-dfreq+0.5)/(dfreq+0.5))+1)
        _score = _idf * ((dtfreq*(k+1))/(dtfreq+k))
        
        return _score

    N = Edgelist_class.d_class.unique().size
    my_output = np.vectorize(bm25)(N,Edgelist_class.ctfreq, Edgelist_class.cfreq, 1.2)

    Edgelist_class['bm25'] = my_output
    
    return Edgelist_class

def make_base_dict(Edgelist_class, qs=0.005, qt=0.5, ns=100, nm=100):   

    # 불용어(stop)은 class별로 하위 q%, 주제어(topic)은 class별로 상위 q%의 단어들을 filtering
    def make_worddict(_data, q = 0.001, mode='stop'):
        if mode == 'stop':
            _und = np.quantile(_data.bm25.values, q)
            _word = _data[_data.bm25 < _und].word.tolist()
        elif mode == 'topic':
            _und = np.quantile(_data.bm25.values, 1-q)
            _word = _data[_data.bm25 >= _und].word.tolist()
        return _word

    stop_word = Edgelist_class.groupby('d_class').apply(lambda x:make_worddict(x, qs, mode='stop'))
    topic_word = Edgelist_class.groupby('d_class').apply(lambda x:make_worddict(x, qt, mode='topic'))    

    # 각 class 집단에서 특정 수 이상 불용어로 등장한 단어들만 불용어로 간주, max(ns), max(nm) = len(class)
    stop_all_c = list(itertools.chain(*stop_word))
    stop_all_c = Counter(stop_all_c)
    stop_all = [i for i,v in stop_all_c.items() if v > ns] 

    topic_all_c = list(set(list(itertools.chain(*topic_word))))
    topic_all = [i for i in topic_all_c if len(i) > 1]        # 한글자 제거
    # word_all = list(stop_all)+list(topic_all)

    # # 각 class 집단에서 특정 수 이상 연결어로 등장한 단어들만 연결어로 간주, 그렇지 않은 단어들은 주제어로 이동
    # middle_all_c = Edgelist_class[~Edgelist_class.word.isin(word_all)].word.tolist()
    # middle_all_c = Counter(middle_all_c)
    # middle_all = [i for i,v in middle_all_c.items() if v > nm]    
    # topic_all = topic_all + [i for i,v in middle_all_c.items() if v <= nm]
    middle_all = []
    return middle_all, topic_all, stop_all

def preprocessing_create_edgelist_term(word_dataframe):
    nounmaker = LRNounExtractor_v2(verbose=False)
    _pos = nounmaker._pos_features
    _neg = nounmaker._neg_features
    features_set = sorted(_pos.union(_neg), key=len, reverse=True)
    features_delete_mask = (word_dataframe['word'].isin(features_set))

    regex_word_delete_mask = (word_dataframe['word'].map(lambda x: delete_word_after_making_userdict(x)))

    return word_dataframe.loc[~regex_word_delete_mask & ~features_delete_mask]

"""
# middle_all과 topic_all은 list 형식의 단어집합

middle_all = [학습, 전달, ...]
topic_all = [인공지능, 컴퓨터, ...]

# bridge_word도 list 형식의 단어집합

bridge_word = [정제, 검수, ...]

"""

def extract_recommend_word(Edgelist):

    word_list = Edgelist.copy()
    word_list['d_class'] = 'S'
    word_list = make_rawdata_class(word_list)
    word_list = word_list.drop_duplicates('word').sort_values('bm25',ascending = False).head(100).word.tolist()

    return word_list


def extract_bridge_word(EgoEdgelist, middle_all, w_limit = 1000):
    
    # 직접 연결된 관계만 고려하기 위하여 탐색범위를 좌우 1로 설정(window_size = 1)
    # pmi_mean은 기댓값의 정의와 동일하게, p_rc * pmi(두 단어가 함께 등장할 확률 * 두 단어 관계가 가지는 pmi 값)
    # 행렬곱 한 후 행 기준으로 더하면 해당 단어의 평균 PMI값을 계산할 수 있다.
    # alpha parameter는 기준 단어와 함께 등장하는 단어의 확률을 제한하는 역할을 한다. (자세한 내용은 PMI수식 참조)

    EgoEdgelist = EgoEdgelist[EgoEdgelist.col_type == 'content']
    pmi_mat, p_rc, w_dict = make_pmi(EgoEdgelist, window_size=1, alpha=0.000, sample_n = False, min_pmi = 0)
    pmi_mean = pmi_mat.dot(p_rc.T) 
    
    N = np.array([len(n) for n in np.split(pmi_mean.indices, pmi_mean.indptr)[1:-1]]) # 행별로 PMI값이 존재하는 원소의 수 array
    pmi_mean = pmi_mean.sum(axis=1).A1 # 행 기준으로 각 단어의 평균 PMI값 계산
    
    w_dict_inv = {v:i for i, v in w_dict.items()}

    pmi_mean_c = pmi_mean+((N)/(N.max())) # 많은 단어들과 함께 등장하는 단어에 가중치를 준다. 범위는 0~1 이다.
    bridge_word = [w_dict_inv[i] for i in pmi_mean_c.argsort()[::-1][:w_limit] if w_dict_inv[i] in middle_all] # 최대 1000개의 단어까지 연결어로 제안한다. 
    
    return bridge_word


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
    from terms.userdict_utils import delete_word_after_making_userdict
    import timeit
    from config.constants import DOCUMENT_DB_NAMES

    start_time = timeit.default_timer()
    target_db = 'test_db'
    we_edge = SnaEdgelist()
    all_edgelist = we_edge.get_all_edgelist(target_db=target_db)
    Edgelist_class = make_rawdata_class(all_edgelist)
    Edgelist_class.to_parquet(f'/data/temp_parquet/{target_db}_Edgelist_class.parquet')
    
#     for target_db in DOCUMENT_DB_NAMES:
#         we_edge = SnaEdgelist()
#         all_edgelist = we_edge.get_all_edgelist(target_db=target_db)
#         Edgelist_class = make_rawdata_class(all_edgelist)
#         middle_all, topic_all, stop_all = make_base_dict(Edgelist_class, qs=0.001, nm=50)
#         topic_df = pd.DataFrame({'topic_word':topic_all})
#         middle_df = pd.DataFrame({'middle_word':middle_all})
#         stop_df  = pd.DataFrame({'stop_word':stop_all})
        
#         topic_df.to_parquet(f'/data/temp_parquet/{target_db}_topic_df.parquet')
#         middle_df.to_parquet(f'/data/temp_parquet/{target_db}_middle_df.parquet')
#         stop_df.to_parquet(f'/data/temp_parquet/{target_db}_stop_df.parquet')
    print(timeit.default_timer()-start_time)
#     # print(topic_all[0:100])