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
    http_auth=(user_passw[0], user_passw[1]), timeout=120
)

INDEX = 'covid19misinfo-2020-04'

def practice_funcs():

  # running 10 hit query
  query_body = {
    "query": {
      "bool" : {
        "must" : {
          "exists": {
                'field': 'entities.Tweet.retweeted_status'
          }
        },
        "filter": {
          "range": {"entities.Tweet.created_at": {"gte": "Wed Apr 15 00:00:00 +0000 2020","lte": "Wed Apr 15 23:59:59 +0000 2020"} }
        }
        #"minimum_should_match" : 1,
        #"boost" : 1.0
      }
    }
  }

  result = es.search(index = INDEX, body = query_body)


  # learn about entities in tweet
  entry_entities = result['hits']['hits'][0]['_source']['entities']
  for field in entry_entities:
    print(field, ' -> ', entry_entities[field])

  # learn about tweet makeup
  first_tweet = result['hits']['hits'][0]['_source']['entities']['Tweet']
  first_tweet_df = pd.DataFrame(result['hits']['hits'][0])
  for section in first_tweet:
    for field in section:
      print(field, ' -> ', section[field])

  # e.g. to access tweet quote_count, use entities.Tweet.quote_count

  # find only the retweets 

  # objective: find the top ten retweeted tweets in a certain day in April
  #           - unfortunately retweet counts are not possible to get for individual tweets in the database as it is a data stream
  #           - so i can select a certain day, organise retweets by their IDS, find out how many unique retweet IDS there are
  #           - could use a dictionary or dataframe, with counter for each unique ID

  # to get id of a original tweet from retweet, result['hits']['hits'][0]['_source']['entities']['Tweet'][0]['retweeted_status']['id_str']

  rt_tweet = result['hits']['hits'][0]['_source']['entities']['Tweet']['retweeted_status']
  print(result['hits']['hits'][0]['_source']['entities']['Tweet'][0]['retweeted_status']['id_str'])


  result = es.search(INDEX, body = query_body, size = 10000)
  print ("total hits using 'size' param:", len(result["hits"]["hits"]))


# function to return a tweet from its ID 
def print_tweet_body(id):

  query_body = {
    "query": {
      "match" : {
        "entities.Tweet.id" : id
      }
    }
  }

  result = es.search(index = INDEX, body = query_body)
  tweet = result['hits']['hits'][0]['_source']['entities']['Tweet']
  for section in tweet:
    for field in section:
      print(field, ' -> ', section[field])

# separate function to select certain day, process RTs
def count_rts(query_size) :
  unique_id_store = dict()

  query_body = { 
    "query": {
      "bool" : {
        "must" : {
          "exists": {
                'field': 'entities.Tweet.retweeted_status'
          }
        },
        "filter": {
          "range": {"entities.Tweet.created_at": {"gte": "Wed Apr 15 00:00:00 +0000 2020","lte": "Wed Apr 15 23:59:59 +0000 2020"} }
        }
        #"minimum_should_match" : 1,
        #"boost" : 1.0
      }
    }
  }

  result = es.search(index = 'covid19misinfo-2020-04', body = query_body, size = query_size)
  retweets = result['hits']['hits']

  for tweet in retweets :
    original_id = tweet['_source']['entities']['Tweet'][0]['retweeted_status']['id_str']
    if original_id in unique_id_store:
      unique_id_store[original_id] += 1
    else:
      unique_id_store[original_id] = 0

  
  max_key = max(unique_id_store, key=unique_id_store.get)
  print(f'largest retweeted tweet: {max_key}, with {unique_id_store[max_key]} retweets')
  print(f'tweet body:')
  print_tweet_body(max_key)

  return unique_id_store

count_rts(10000)


