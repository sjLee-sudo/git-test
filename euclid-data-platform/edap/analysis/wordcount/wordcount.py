import datetime
import pandas as pd
import numpy as np
import copy
print(__name__)
if __name__ !='__main__':
    print('test')
    from analysis.edgelist.kistep_edgelist import WordCountEdgelist


def sunburst_treemap_chart(main_word_df):
    sunburst_chart_df = main_word_df[["doc_id","term","pyear","section","subclass"]]
    if isinstance(sunburst_chart_df, pd.Series):
        sunburst_chart_df = sunburst_chart_df.to_frame().transpose()
    sunburst_chart_df = sunburst_chart_df.drop_duplicates(["term","doc_id"])
    word_sum = sunburst_chart_df.groupby("term")["doc_id"].count()
    word_sum = word_sum.reset_index()
    word_sum["parents"] = ""
    word_sum = word_sum.transpose()
    word_sum = word_sum.rename({"parents":"parents","term":"labels","doc_id":"values"})
    word_section_sum = sunburst_chart_df.groupby(["term","section"])["doc_id"].count()
    word_section_sum = word_section_sum.reset_index().transpose()
    word_section_sum = word_section_sum.rename(index={"term":"parents","section":"labels","doc_id":"values"})
    section_subclass_sum = sunburst_chart_df.groupby(["section","subclass"])["doc_id"].count()
    section_subclass_sum = section_subclass_sum.reset_index().transpose()
    section_subclass_sum = section_subclass_sum.rename(index={"section":"parents","subclass":"labels","doc_id":"values"})
    result = pd.concat([word_sum,word_section_sum,section_subclass_sum], join="outer", axis=1)
    parents_list = result.loc["parents"].to_list()
    labels_list  = result.loc["labels"].to_list()
    values_list =  result.loc["values"].to_list()
    sunburst_chart_result = [{"parents":parents_list, "labels":labels_list,"values":values_list,"comment":["labels 수와 values 수는 일치해야 함"]}]
    treemap_chart_result =  sunburst_chart_result
    return sunburst_chart_result, treemap_chart_result

def section_word_cumu_chart(main_word_df):
            
    section_word_cumu_df = main_word_df[["doc_id","pyear","subclass","section"]]
    if isinstance(section_word_cumu_df, pd.Series):
        section_word_cumu_df = section_word_cumu_df.to_frame().transpose()
    section_word_cumu_df = section_word_cumu_df.drop_duplicates()
    section_word_cumu_df = section_word_cumu_df.sort_values("pyear")
    word_sum = section_word_cumu_df.groupby(["pyear","section"])["doc_id"].count()
    section_word_cumu_df = section_word_cumu_df.set_index(["pyear","section"])
    section_word_cumu_df["word_cnt_sum"] = word_sum
    section_word_cumu_df = section_word_cumu_df.reset_index()
    section_word_cumu_df = section_word_cumu_df.drop_duplicates(["pyear","section"])
    section_word_cumu_result = [{"x": section_word_cumu_df.pyear, "y":section_word_cumu_df.word_cnt_sum, "name": section_word_cumu_df.section}]
    return section_word_cumu_result

def word_cumu_chart(main_word_df):
    cumu_df = main_word_df[["pyear", "doc_id"]]
    if isinstance(cumu_df, pd.Series):
        cumu_df = cumu_df.to_frame().transpose()
    cumu_df = cumu_df.drop_duplicates()        
    cumu_df = cumu_df.sort_values("pyear")
    cumu_df = cumu_df.set_index(["pyear"])
    word_cnt_sum = cumu_df.groupby(cumu_df.index)["doc_id"].count()
    cumu_df["word_cnt"] = word_cnt_sum
    cumu_df = cumu_df.reset_index()
    cumu_df = cumu_df.drop_duplicates("pyear")
    word_cumu_result = [{"x": cumu_df.pyear, "y": cumu_df.word_cnt}]
    return word_cumu_result


def word_curve_chart(main_word_df, sort):
    curve_df = main_word_df[["pyear","doc_id"]]
    curve_df = curve_df.drop_duplicates()
    word_cnt = curve_df.groupby("pyear")["doc_id"].count()
    curve_df = curve_df.set_index("pyear")
    curve_df["word_cnt"] = word_cnt
    curve_df = curve_df.reset_index().drop_duplicates("pyear").sort_values("pyear")
    curve_df["sort"] = sort
    word_curve_result_list = []
    for idx in curve_df.index:
        word_curve_result_list.append({"date" : curve_df.loc[idx,"pyear"], "sort": curve_df.loc[idx,"sort"], "value": curve_df.loc[idx,"word_cnt"]})
    return word_curve_result_list

def count_base_chart(main_word_df):
    cumsum_df = main_word_df[["doc_id","pyear","subclass","section"]]
    sc_pyear_sum = cumsum_df.groupby(["subclass","pyear"])["doc_id"].count()
    cumsum_df = cumsum_df.set_index(["subclass","pyear"])
    cumsum_df["sc_pyear_sum"] = sc_pyear_sum
    cumsum_df = cumsum_df.reset_index()[["pyear","section","subclass","sc_pyear_sum"]].drop_duplicates().sort_values(["subclass","pyear"])
    sc_cumsum = cumsum_df.groupby("subclass")["sc_pyear_sum"].cumsum()
    cumsum_df["sc_cumsum"] = sc_cumsum
    count_base_result = [{ "S":cumsum_df.section,"SC":cumsum_df.subclass, "Y": cumsum_df.pyear, "C":cumsum_df.sc_cumsum,"comment":["S, SC, Y 분류별로 C를 집계"] }]
    return count_base_result

def make_cloud_chart(related_word_df):
    cloud_doc_id_df = related_word_df[["term","doc_id"]]
    cloud_chart_list = []
    if cloud_doc_id_df.empty:
        return cloud_chart_list
    cloud_doc_id_df = cloud_doc_id_df.drop_duplicates()
    word_cnt = cloud_doc_id_df.groupby("term").doc_id.count()
    cloud_doc_id_df = cloud_doc_id_df.set_index("term")
    cloud_doc_id_df["word_cnt"] = word_cnt
    _max = 12
    _wordsum_max = cloud_doc_id_df.word_cnt.max()
    _denom = _wordsum_max / _max
    cloud_doc_id_df = cloud_doc_id_df.sort_values("word_cnt",ascending=False)
    cloud_doc_id_df["word_cnt"] = (np.ceil(cloud_doc_id_df.word_cnt / _denom)).astype(int)
    for word in cloud_doc_id_df.index.unique():
        sub_list = []
        sub_list.append(word)
        try:
            word_cnt = cloud_doc_id_df.loc[word,"word_cnt"]
            if isinstance(word_cnt, pd.Series):
                sub_list.append(word_cnt[0])
            else:
                sub_list.append(word_cnt)
        except KeyError:
            sub_list.append(0)
        try:
            doc_id_list = []
            doc_id_record = cloud_doc_id_df.loc[word,"doc_id"]
            if isinstance( doc_id_record, pd.Series):
                doc_id_list = doc_id_record.tolist()
            else:
                doc_id_list = [doc_id_record]
            sub_list.append(doc_id_list)
        except KeyError:
            sub_list.append(doc_id_list)

        cloud_chart_list.append(sub_list)
    cloud_chart_result = [cloud_chart_list]
    return cloud_chart_result


def make_bar_and_donut_chart(word_list, word_cnt_sum_df):
    
    x_list = [] 
    y_list = []
    for word in word_list:
        x_list.append(word)
        try:
            word_cnt = word_cnt_sum_df.loc[word,"word_cnt"]
            if isinstance(word_cnt, pd.Series):
                y_list.append(word_cnt[0])
            else:
                y_list.append(word_cnt)
        except KeyError:
            y_list.append(0)
    
    bar_chart_result = [{"x":x_list,"y":y_list}]
    donut_chart_result = [{"labels": x_list, "values": y_list}]
    return bar_chart_result,donut_chart_result 

    
def make_word_group_bubble_chart(related_word_df):
    bubble_df = related_word_df[["term","pyear","doc_id"]]
    bubble_df = bubble_df.sort_values(["term","pyear"])
    word_cnt_sum_series = bubble_df.groupby(["term","pyear"])["doc_id"].count()
    bubble_df = bubble_df.set_index(["term","pyear"])
    bubble_df["word_cnt"] = word_cnt_sum_series
    bubble_df = bubble_df.reset_index().drop_duplicates(["term","pyear"])[["term","pyear","word_cnt"]]
    bubble_df["cumsum"] = bubble_df.groupby(["term"])["word_cnt"].cumsum()
    bubble_df = bubble_df.set_index("term")
    bubble_chart_result = []
    max_rate = 0
    max_word_cnt = 0
    min_rate = 0
    for word in bubble_df.index.unique():
        tmp_df = bubble_df.loc[word]
        if isinstance(tmp_df, pd.Series):
            tmp_df = tmp_df.to_frame().transpose()
        tmp_df = tmp_df.reset_index()
        for idx, row in tmp_df.iterrows():
            rate = 0.00
            if idx > 0:
                rate = round(((tmp_df.loc[idx,"word_cnt"]-tmp_df.loc[idx-1,"word_cnt"])/tmp_df.loc[idx-1,"word_cnt"])*100,2)
                # if rate < 0:
                    # rate = round(-((rate*-1) ** (1/1.5) ),3)
                # else:
                # rate = round(rate ** (1/1.5),3)
                if rate >= max_rate:
                    max_rate = rate
                if rate <= min_rate:
                    min_rate = rate
            word_cnt = tmp_df.loc[idx, "word_cnt"]
            # word_cnt = round(word_cnt **(1/1.5),3)
            cumsum = tmp_df.loc[idx, "cumsum"]
            # cumsum = round(cumsum ** (1/1.5),3)
            if cumsum >= max_word_cnt:
                    max_word_cnt = cumsum
            pyear = tmp_df.loc[idx, "pyear"]
            bubble_chart_result.append({"group":word, "id":word, "x": cumsum ,"y": rate, "size": word_cnt, "year": pyear})
    max_word_cnt = max_word_cnt  * 1.1
    max_rate = max_rate * 1.1
    min_rate = min_rate * 1.1
    return bubble_chart_result, min_rate, max_rate, max_word_cnt
    
def make_class_group_bubble_chart(word_list, word_cnt_sum_df):
    bubble_df = word_cnt_sum_df[["pyear","doc_id","section"]]
    bubble_df = bubble_df.reset_index().sort_values(["section","pyear"])
    bubble_df = bubble_df[bubble_df.word.isin(word_list)]
    word_cnt_sum_series = bubble_df.groupby(["section","pyear"])["doc_id"].count()
    bubble_df = bubble_df.set_index(["section","pyear"])
    bubble_df["word_cnt"] = word_cnt_sum_series
    bubble_df = bubble_df.reset_index().drop_duplicates(["term","section","pyear"])
    bubble_df["cumsum"] = bubble_df.groupby(["section","term"])["word_cnt"].cumsum()
    bubble_df = bubble_df.sort_values(["section", "term"])
    bubble_df = bubble_df.set_index(["section", "term"])
    bubble_chart_result = []
    max_rate = 0
    max_word_cnt = 0
    for section_word in bubble_df.index.unique():
        tmp_df = bubble_df.loc[section_word]
        if isinstance(tmp_df, pd.Series):
            tmp_df = tmp_df.to_frame().transpose()
        tmp_df = tmp_df.reset_index()
        for idx, row in tmp_df.iterrows():
            rate = 0.00
            if idx > 0:
                rate = round(((tmp_df.loc[idx,"cumsum"]-tmp_df.loc[idx-1,"cumsum"])/tmp_df.loc[idx-1,"cumsum"]),2)
                if rate >= max_rate:
                    max_rate = rate
            word_cnt = tmp_df.loc[idx, "word_cnt"]
            if word_cnt >= max_word_cnt:
                max_word_cnt = word_cnt
            cumsum = tmp_df.loc[idx, "cumsum"]
            pyear = tmp_df.loc[idx, "pyear"]
            bubble_chart_result.append({"group":section_word[0], "id":section_word[1], "x": word_cnt ,"y": rate, "size": cumsum, "year": pyear})
    return bubble_chart_result, max_rate, max_word_cnt



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
    from analysis.edgelist.kistep_edgelist import WordCountEdgelist
    from db_manager.managers import ElasticQuery
    
    import timeit
    # start_time = timeit.default_timer()
    # target_db = 'docu_patent'
    # search_text = '인공지능 카테고리 자율주행 자동차'
    # es_query = ElasticQuery(target_db=target_db)
    # search_tokens = es_query.tokenize_text(search_text)
    # print(search_tokens)

    # wc_edge = WordCountEdgelist()
    # wc_ego_edgelist = wc_edge.get_wc_ego_edgelist(target_db=target_db, search_text=search_text)
    # print(wc_ego_edgelist.head())
    # print(wc_ego_edgelist.shape)
    # print(timeit.default_timer()-start_time)