import json
import requests
from elasticsearch import Elasticsearch
from collections import namedtuple

es = Elasticsearch([{
    'host': 'localhost',
    'port': 9200
}])

request_body = {
    "mappings": {
        "book": {
            "properties": {
				'id': {'type': 'integer'},
                'genre': {'type': 'text'},
                'title': {'type': 'text'},
                'author': {'type': 'text'},
                'publisher': {'type': 'text'},
                'edition': {'type': 'integer'}

            }
        }
    },

    "settings": {
        "number_of_shards": 5,
        "number_of_replicas": 1
    }
}

es.indices.create(index='book_data', body=request_body)

data = json.loads(open("./book.json", encoding="utf8").read())

i = 0

while i < 1000:
    es.index(index='book_data', doc_type='book', id=i, body=data[i])
    i += 1

print("Data inserted with mapping......")

