import os
from re import X
import requests
import networkx as nx

from elasticsearch import Elasticsearch

auth_token = os.environ.get('ELASTIC_TOKEN')

# split auth token from form user:pass into list
user_passw = auth_token.split(':')[0]
user, passw = user_passw[0], user_passw[1]


INDEX = 'covid19misinfo-2020-04'
TIMEOUT = 360

es = Elasticsearch(
    'http://gateservice10.dcs.shef.ac.uk:9300',
    http_auth=(user_passw[0], user_passw[1]), timeout=TIMEOUT
)

# twitter connecting stuff if needed

# need to form twitter authorization headers
def connect_to_twitter():
    # secure version to get bearer token from environment variable
    bearer_token = os.environ.get("BEARER_TOKEN")
    return {"Authorization": "Bearer {}".format(bearer_token)}
headers = connect_to_twitter()

def make_request(headers, url):
    #url = "https://api.twitter.com/2/tweets/1491217291669049344"
    return requests.request("GET", url, headers=headers).json()

class StoreRetweets:

  def __init__(self, start_date, end_date, index):
    self.tweets = ''
    self.index = index
    self.quoteG = nx.DiGraph()
    self.start_date = start_date
    self.end_date = end_date

    # query body to find quote tweets within dataset
    self.quoted_only = { 
      "query": {
        "bool" : {
          "must" : {
            "exists": {
                  'field': 'entities.Tweet.quoted_status'
            }
          },
            "filter": {
              "range": {"entities.Tweet.created_at": {"gte": self.start_date,"lte": self.end_date} }
            }
        }
      }
    }

  # function to check if tweet ID is present in the ES database
  def is_tweet_present(self, tweet_id):
    query_body = { 
      "query": {
        "bool" : {
          "must" : {
            "term": {"entities.Tweet.id" : tweet_id }
          },
          "filter": {
            "range": {"entities.Tweet.created_at": {"gte": self.start_date,"lte": self.end_date} }
          }
        }
      }
    }

    result = es.search(index= self.index, body = query_body, size = 1)

    if len(result['hits']['hits']) > 0:
      return True
    else:
      return False

  # function will get quote tweets from the elasticsearch dataset and link them back to their top original tweet, building network along way
  def pull_quotes(self, query_size):

    result = es.search(index= self.index, body = self.quoted_only, size = query_size)
    quotes = result['hits']['hits']
    for quote in quotes:
      quote_body = quote['_source']['entities']['Tweet'][0]
      quote_id = quote_body['id_str']

      #add quote tweet as node into graph - networkx ignores double entry

      #checking to see whether to add the edge between original and quoted tweet : 
      #     1- check whether original ID is in the elasticsearch database to start with
      #     2- then can add an edge between the original and quoted tweet
      #     3- check whether original tweet itself is in the elasticsearch database
      #     4- return to step 1, until step 3 not satisfied
      
      original_body = quote_body['quoted_status']
      original_id = original_body['id_str']
      
      if self.is_tweet_present(original_id):
        self.quoteG.add_node(quote_id)
        self.quoteG.add_node(original_id)
        self.quoteG.add_edge(quote_id, original_id)
        #print('original id present in DB')

      #if original_body['is_quote_status']:

  # improved function to query quote tweet orginal id's further
  def pull_quote_chain(self, query_size):

    result = es.search(index= self.index, body = self.quoted_only, size = query_size)
    quotes = result['hits']['hits']

    for quote in quotes:
      quote_body = quote['_source']['entities']['Tweet'][0]
      quote_text= quote['_source']['entities']['Tweet'][0]
      quote_id = quote_body['id_str']

      original_body = quote_body['quoted_status']
      original_id = original_body['id_str']

      # entities.Tweet.quoted_status.quoted_status_id_str

      if self.is_tweet_present(original_id):
        self.quoteG.add_node(quote_id)
        self.quoteG.add_node(original_id)
        self.quoteG.add_edge(original_id, quote_id)

      # if a further quote object found, continue linking
      if 'quoted_status_id_str' in original_body.keys():
        further_id = original_body['quoted_status_id_str']
        self.quoteG.add_node(further_id)
        self.quoteG.add_edge(further_id, original_id)

  def pull_quote_period():
    x





      


    




