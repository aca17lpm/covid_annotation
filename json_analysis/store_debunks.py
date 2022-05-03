import os
from re import X
import requests
import asyncio
import json
import networkx as nx
import tweepy

bearer_token = os.environ.get("BEARER_TOKEN")
client = tweepy.Client(bearer_token)

class StoreDebunks:

  def __init__(self, debunk_path):
    self.tweet_path = debunk_path
    self.tweetG = nx.DiGraph()
    self.unique_id_store = dict()

  def pull_tweets(self):

    with open(self.tweet_path, encoding="utf-8") as f:
      tweets = json.load(f)

      for tweet in tweets:
        if tweet['tweet_label'] == 'DEBUNK':
          debunk_id = tweet['tweet_id']
          print(f'Finding quotes of tweet with id: {debunk_id}')
          print(client.get_quote_tweets(debunk_id))

print(client.get_tweet(1519781958578016256))
          


    





      


    




