INDEX_TEMPLATE = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 1,
    },
    "mappings": {
        "properties": {
            "unique_id": {"type": "keyword"},
            "input_data": {"type": "object"},
            "stored_data":{"type":"object"},
            "remark": {
                "type": "text", 
                "fields": {"keyword": {"type": "keyword"}
                }
            },
            "timestamp": {"type": "date"}
        }
    }
}