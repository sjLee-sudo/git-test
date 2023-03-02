INDEX_TEMPLATE = {
    "settings": {
        "number_of_shards": 10,
        "number_of_replicas": 0,
        "index": {
            "analysis": {
                "analyzer": {
                    "custom_noun_analyzer": {
                        "char_filter": ["html_strip","remove_cr"],
                        "tokenizer": "custom_nori_tokenizer",
                        "filter": [
                            "trim",
                            "nori_readingform",
                            "noun_posfilter",
                            "lowercase",
                            "representative_filter",
                            "custom_stop_filter"
                        ],
                        "type": "custom"
                    },
                    "decompound_none_mode_analyzer": {
                        "char_filter": ["html_strip","remove_cr"],
                        "tokenizer": "none_mode_nori_tokenizer",
                        "filter": [
                            "trim",
                            "nori_readingform",
                            "noun_posfilter",
                            "lowercase",
                            "representative_filter",
                            "custom_stop_filter"
                        ],
                        "type": "custom"
                    },
                    "custom_search_analyzer": {
                        "char_filter": ["html_strip"],
                        "tokenizer": "custom_nori_tokenizer",
                        "filter": [
                            "trim",
                            "nori_readingform",
                            "noun_posfilter",
                            "lowercase",
                            "representative_filter",
                            "synonym_filter",
                            "custom_stop_filter"
                        ],
                        "type": "custom"
                    },
                    "english_analyzer":{
                        "tokenizer": "letter",
                        "filter": [
                            "trim",
                            "lowercase",
                        ]
                    },
                    "comma_pattern_analyzer":{
                        "char_filter": ["remove_cr"],
                        "tokenizer": "comma_pattern_tokenizer",
                        "filter": ["trim","lowercase"]
                    }
                },
                "char_filter":{
                    "remove_cr":{
                        "type": "pattern_replace",
                        "pattern": "[\r\f\n]+",
                        "replacement": " "
                    }
                },
                "filter": {
                    "noun_posfilter": {
                        "type": "nori_part_of_speech",
                        "stoptags": [
                            "E",
                            "IC",
                            "J",
                            "MAG",
                            "MAJ",
                            "MM",
                            "SP",
                            "SSC",
                            "SSO",
                            "SC",
                            "SE",
                            "XPN",
                            "XSA",
                            "XSN",
                            "XSV",
                            "NR",
                            "SSO",
                            "SY",
                            "SN",
                            "VV",
                            "NNB",
                            "SF",
                            "VCP",
                            "VA",
                            "VX",
                            "VCN",
                            "VSV",
                            "XR",
                            "NP"
                        ]
                    },
                    "synonym_filter": {
                        "type": "synonym",
                        "synonyms_path": "userdict/limenet_analysis/synonym.txt",
                        "lenient": "true",
                    },
                    "representative_filter": {
                        "type": "synonym",
                        "synonyms_path": "userdict/limenet_analysis/representative.txt",
                        "lenient": "true"
                    },
                    "custom_stop_filter": {
                        "type": "stop",
                        "stopwords_path": "userdict/limenet_analysis/stopword.txt"
                    }
                },
                "tokenizer": {
                    "custom_nori_tokenizer": {
                        "user_dictionary": "userdict/limenet_analysis/terminology.txt",
                        "decompound_mode": "mixed",
                        "type": "nori_tokenizer",
                        "discard_punctuation": "true"
                    },
                    "comma_pattern_tokenizer":{
                        "type": "pattern",
                        "pattern": ","
                    },
                    "none_mode_nori_tokenizer":{
                        "user_dictionary": "userdict/limenet_analysis/terminology.txt",
                        "decompound_mode": "none",
                        "type": "nori_tokenizer",
                        "discard_punctuation": "true"
                    }
                }
            }
        }
    },
    "mappings": {
        "_routing": {"required": "true"},
        "properties": {
            "doc_id": {"type": "keyword"},
            "pjt_id": {"type": "keyword"},
            "prog_nm": {
                "type": "text",
                "analyzer": "custom_noun_analyzer",
                "search_analyzer": "custom_search_analyzer",
                "fields": {"keyword": {"type": "keyword", "ignore_above":3000}}
            },
            "prog_mstr_nm": {"type":"keyword"},
            "kor_pjt_nm": {
                "type": "text",
                "analyzer": "custom_noun_analyzer",
                "search_analyzer": "custom_search_analyzer",
                "fields": {"keyword": {"type": "keyword", "ignore_above":3000}}
            },
            "stan_yr":{
                "type": "date",
                "format": "yyyy||epoch_millis"
            },
            "spclty_org_nm": {"type":"keyword"},
            "rndco_tot_amt": {"type": "long"},
            "tot_rsch_start_dt": {"type": "date", "ignore_malformed": "true", "format": "yyyy-MM-dd HH:mm:ss||yyyy-MM-dd||epoch_millis"},
            "tot_rsch_end_dt": {"type": "date", "ignore_malformed": "true", "format": "yyyy-MM-dd HH:mm:ss||yyyy-MM-dd||epoch_millis"},
            "dtpr_prog_nm":{
                "type": "text",
                "analyzer": "custom_noun_analyzer",
                "search_analyzer": "custom_search_analyzer",
                "fields": {"keyword": {"type": "keyword", "ignore_above":3000}}
            },
            "rsch_goal_abstract": {
                "type": "text",
                "analyzer": "custom_noun_analyzer",
                "search_analyzer": "custom_search_analyzer"
            },
            "rsch_abstract": {
                "type": "text",
                "analyzer": "custom_noun_analyzer",
                "search_analyzer": "custom_search_analyzer"
            },
            "exp_efct_abstract": {
                "type": "text",
                "analyzer": "custom_noun_analyzer",
                "search_analyzer": "custom_search_analyzer"
            },
            "kor_kywd":{
                "type":"text",
                "analyzer": "comma_pattern_analyzer"
            },
            "eng_kywd":{
                "type":"text",
                "analyzer": "comma_pattern_analyzer"
            },
            "nat_strt_tech_cd": {"type": "keyword", "ignore_above":100},
            "doc_section": {"type": "keyword", "ignore_above": 10},
            "doc_class": {"type": "keyword" ,"ignore_above":30},
            "doc_subclass": {"type": "keyword" ,"ignore_above":10},
            "analysis_target_text":{
                            "type":"text",
                            "analyzer": "decompound_none_mode_analyzer"
            },
            "src_org_id":{"type":"keyword"},
            "pre_pjt_id":{"type":"keyword"},
            "eng_pjt_nm":{
                "type":"text",
                "analyzer": "english_analyzer",
                "fields": {"keyword": {"type": "keyword", "ignore_above":3000}}
            },
            "prog_mstr_cd":{"type":"keyword"},
            "pjt_mgnt_org_cd":{"type":"keyword"},
            "pjt_prfrm_org_cd":{"type":"keyword"},
            "pjt_prfrm_org_nm":{"type":"keyword"},
            "appl_area_cls_cd":{"type":"keyword"},
            "appl_area_cls_nm":{"type":"keyword"},
            "rsch_area_cls_cd":{"type":"keyword"},
            "rsch_area_cls_nm":{"type":"keyword"},
            "t6tech_cd":{"type":"keyword"},
            "org_bdgt_prog_cd":{"type":"keyword"},
            "pjt_prgs_stat_slct":{"type":"keyword"},
            "pjt_prgs_stat_slct_nm":{"type":"keyword"},
            "prtcp_mp":{
                "type": "keyword",
                "index": "false"
            },
            "ipr":{
                "type": "keyword",
                "index": "false"
            },
            "paper":{
                "type": "keyword",
                "index": "false",
            },
            "secret_pjt_yn":{"type":"keyword"},
            "secret_pjt_canc_ym":{"type":"keyword"},
            "timestamp": {"type": "date"}
        }
    }
}