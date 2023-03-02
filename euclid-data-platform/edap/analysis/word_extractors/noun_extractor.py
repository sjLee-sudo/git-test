import os
import pandas as pd
from soynlp.userdict import NounUserdictMaker
from soynlp.noun import LRNounExtractor_v2


dirname = '/'.join(os.path.abspath(__file__).replace('\\', '/').split('/')[:-1])
DEFAULT_PREDICT_HEADERS = dirname+'/noun_predict_model/noun_predictor_patent'

def create_noun_compound_df_from_text_list(text_list):

    noun_extractor = LRNounExtractor_v2(max_left_length=15, max_right_length=6, predictor_headers=DEFAULT_PREDICT_HEADERS,
            verbose=True, min_num_of_features=2, max_frequency_when_noun_is_eojeol=30,
            eojeol_counter_filtering_checkpoint=500000,
            extract_compound=False, extract_pos_feature=False, extract_determiner=False,
            ensure_normalized=True, postprocessing=['detaching_features','ignore_features','custom_ignore_NJ'])
            # default postprocessing = ['detaching_features','ignore_features','ignore_NJ']

    userdict_maker = NounUserdictMaker(noun_extractor=noun_extractor, priority_terms_path=None, stop_terms_path=None,
                    min_term_len=2, min_noun_score=0.5, min_noun_frequency=10, remove_only_english=True, remove_only_number=True,remove_not_korean=True,
                    verbose=True, ensure_normalized=True)

    # 문장리스트에서 사용자 사전 생성 시 (parameter: list, return: pandas dataframe)
    userdict_dataframe = userdict_maker.create_userdict_from_sentence_list(text_list)
    return userdict_dataframe
