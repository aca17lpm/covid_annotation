import os
import json
import pandas as pd

from elasticsearch import Elasticsearch

auth_token = os.environ.get('ELASTIC_TOKEN')

# split auth token from form user:pass into list
user_passw = auth_token.split(':')[0]
user, passw = user_passw[0], user_passw[1]


es = Elasticsearch(
    'http://gateservice10.dcs.shef.ac.uk:9300',
    http_auth=(user_passw[0], user_passw[1])
)

query_body = {
  'query': {
      'exists': {
        'field': 'entities.Tweet.retweeted_status'
      }
  }
}

result = es.search(index = 'covid19all-2020-04', body = query_body)

# # learn about entities in tweet
# entry_entities = result['hits']['hits'][0]['_source']['entities']
# for field in entry_entities:
#   print(field, ' -> ', entry_entities[field])

# learn about tweet makeup
entry_tweet = result['hits']['hits'][0]['_source']['entities']['Tweet']
for section in entry_tweet:
  for field in section:
    print(field, ' -> ', section[field])

# e.g. to access tweet quote_count, use entities.Tweet.quote_count

# find only the retweets 

# objective: find the top ten retweeted tweets in a certain day in April

