CACHE_INDEX_TEMPLATE = {
    "settings":{
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "analysis_result": {"type": "object", "enabled": "false"},
            "analysis_type": {"type": "keyword"},
            "doc_size": {"type": "keyword"},
            "search_keys": {
                "type": "keyword"
            },
            "target_db": {"type": "keyword"},
            "timestamp": {"type": "date"}
        }
    }
}
