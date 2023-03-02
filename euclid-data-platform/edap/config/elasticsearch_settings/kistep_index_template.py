INDEX_TEMPLATE = {
    "settings": {
        "number_of_shards": 10,
        "number_of_replicas": 0,
        "index": {
            "analysis": {
                "analyzer": {
                    "custom_noun_analyzer": {
                        "char_filter": ["html_strip"],
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
                    "full_filter_analyzer": {
                        "char_filter": ["html_strip"],
                        "tokenizer": "mixed_compound_tokenizer",
                        "filter": [
                            "trim",
                            "nori_readingform",
                            "noun_posfilter",
                            "lowercase",
                            "synonym_filter",
                            "representative_filter",
                            "custom_stop_filter"
                        ],
                        "type": "custom"
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
                        "synonyms_path": "userdict/default/synonym.txt",
                        "lenient": "true",
                    },
                    "representative_filter": {
                        "type": "synonym",
                        "synonyms_path": "userdict/default/representative.txt",
                        "lenient": "true"
                    },
                    "custom_stop_filter": {
                        "type": "stop",
                        "stopwords_path": "userdict/default/stopword.txt"
                    }
                },
                "tokenizer": {
                    "custom_nori_tokenizer": {
                        "user_dictionary": "userdict/default/terminology.txt",
                        "decompound_mode": "none",
                        "type": "nori_tokenizer",
                        "discard_punctuation": "true"
                    },
                    "mixed_compound_tokenizer": {
                        "user_dictionary": "userdict/default/terminology.txt",
                        "decompound_mode": "mixed",
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
            "country_code": {"type": "keyword"},
            "title": {
                "type": "text",
                "analyzer": "custom_noun_analyzer",
                "search_analyzer": "custom_search_analyzer",
                "fields": {"keyword": {"type": "keyword", "ignore_above": 500}}
            },
            "content": {
                "type": "text",
                "analyzer": "custom_noun_analyzer",
                "search_analyzer": "custom_search_analyzer"
            },
            "publication_date": {
                "type": "date",
                "format": "yyyy||yyyy.MM.dd||yyyyMMdd||yyyy-MM-dd||epoch_millis"
            },
            "doc_section": {"type": "keyword", "ignore_above": 100},
            "doc_class": {"type": "keyword", "ignore_above": 100},
            "doc_subclass": {"type": "keyword", "ignore_above": 100},
            "publisher_code": {"type": "keyword"},
            "publisher_name": {
                "type": "text",
                "fields": {"keyword": {"type": "keyword"}}
            },
            "remark": {
                "type": "text", 
                "fields": {"keyword": {"type": "keyword"}
                }
            },
            "timestamp": {"type": "date"},
            "tokenizer_field":{
                "type": "text",
                "analyzer": "full_filter_analyzer",
                "index" : "false"
            }
        }
    }
}