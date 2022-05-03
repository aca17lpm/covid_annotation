
import os
import json
import pandas as pd

from elasticsearch import Elasticsearch

auth_token = os.environ.get('ELASTIC_TOKEN')

# split auth token from form user:pass into list
user_passw = auth_token.split(':')[0]
user, passw = user_passw[0], user_passw[1]


INDEX = 'covid19misinfo-2020-04'
TIMEOUT = 120

es = Elasticsearch(
    'http://gateservice10.dcs.shef.ac.uk:9300',
    http_auth=(user_passw[0], user_passw[1]), timeout=TIMEOUT
)

tweet_id = 1
query_body = { 
  "query": {
    "bool" : {
      "must" : {
        "term": {"entities.Tweet.id" : tweet_id }
      }#,
      # "filter": {
      #   "range": {"entities.Tweet.created_at": {"gte": self.start_date,"lte": self.end_date} }
      # }
    }
  }
}

result = es.search(index= self.index, body = query_body, size = 1)
return result['hits']['hits'][0]