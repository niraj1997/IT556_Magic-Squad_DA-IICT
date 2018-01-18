import csv
from elasticsearch import Elasticsearch, helpers

es = Elasticsearch()
with open('MOCK_DATA.csv','rU') as f:
    reader = csv.DictReader(f)
    helpers.bulk(es, reader, index='book_index', doc_type='bookmap')