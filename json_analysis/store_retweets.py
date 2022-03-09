import os
import json
import pandas as pd
import requests

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

class Store_Retweets:

  def __init__(self):
    self.tweets = ''

  # function will get 
  def pull_and_store_tweets(query_size, start_date, end_date):
    x

