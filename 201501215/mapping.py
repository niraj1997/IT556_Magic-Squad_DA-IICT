import csv
from elasticsearch import Elasticsearch, helpers

es = Elasticsearch()

request_body = {
	    "settings" : {
	        "number_of_shards": 5,
	        "number_of_replicas": 1
	    },

	    'mappings': {
	        'bookmap': {
	            'properties': {
	            	'genre': {'type': 'text'},
	            	'title': {'type': 'text'},
	            	'author': {'type': 'text'},
	                'publisher': {'type': 'text'},
	                'edition': {'type': 'integer'}
	                
	            }}}
	}

#es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

print("creating 'book_index' index...")
es.indices.delete('book_index')
es.indices.create(index = 'book_index', body = request_body)
es.indices.get_alias("*")