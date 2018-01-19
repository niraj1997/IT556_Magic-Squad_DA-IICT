import json
import requests
from elasticsearch import Elasticsearch
from collections import namedtuple

INDEX_NAME='book_data'

es = Elasticsearch([{
    'host': 'localhost',
    'port': 9200
}])
print('\nList All BOOKS ')
res = es.search(index = INDEX_NAME, size=100, body={"query": {"match_all": {}}})
for hit in res['hits']['hits']:
	print(hit["_source"])

print("\nList all the books with title containing word 'Wonderland':")
res = es.search(index=INDEX_NAME, body={"query": {"match": {"title": "Wonderland"}}})
for hit in res['hits']['hits']:
	print(hit["_source"])

print("\nList all the books with Author name 'Janie':")
res = es.search(index=INDEX_NAME, body={"query": {"match": {"author": "Jaine"}}})
for hit in res['hits']['hits']:
	print(hit["_source"])

print("\nList all the books with Genre name 'Guide':")
res = es.search(index=INDEX_NAME, body={"query": {"match": {"genre": "Guide"}}})
for hit in res['hits']['hits']:
	print(hit["_source"])

print('\nList all the books with edition equal to 2:')
res = es.search(index=INDEX_NAME, body={"query": {"match": {"edition": 2}}})
for hit in res['hits']['hits']:
	print(hit["_source"])
