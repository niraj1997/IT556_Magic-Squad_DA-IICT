import csv
from elasticsearch import Elasticsearch, helpers

es = Elasticsearch()

es.search(index="book_index",body={
        "query" :{
                "match" : {
                        'id': '18'
                        }
                
                }
        
        })