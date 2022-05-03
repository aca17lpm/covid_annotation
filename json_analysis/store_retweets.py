import os
import csv
import requests
import pandas as pd
import networkx as nx

from elasticsearch import Elasticsearch
from class_classifier import Classifier
from misinfo_classifier.misinfo_classifier import MisinfoClassifier

auth_token = os.environ.get('ELASTIC_TOKEN')

# split auth token from form user:pass into list
esuser_passw = auth_token.split(':')[0]
esuser, espassw = esuser_passw[0], esuser_passw[1]

TIMEOUT = 360

es = Elasticsearch(
    'http://gateservice10.dcs.shef.ac.uk:9300',
    http_auth=(esuser, espassw), timeout=TIMEOUT
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

  def __init__(self, start_date, end_date, index, query_size):
    self.tweets = ''
    self.index = index
    self.query_size = query_size
    self.quoteG = nx.DiGraph()

    self.unique_id_store = dict()

    self.classification_store = {
      'id' : [],
      'tweet_text' : [],

      # blank fields for classifier
      'tweet_label' : [] ,
      'claim' : [] ,
      'tweet_date' : [] ,
      'num_hashtags' : [] ,
      'hashtag_text' : [] ,
      'has_link' : [] ,
      'polarity' : [] ,
      'subjective' : [] ,
      'pos_words' : [] ,
      'neg_words' : [] ,
      'misinfo_words' : [] ,
      'debunk_words' : [] ,
      'misinfo_hashtags' : [] ,
      'debunk_hashtags' : [] ,
      'hyperpartisan' : [] ,
      'followers' : [],
      'verified': []
    }

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

  # function to add a tweet to be classified as misinfo/debunk,
  # uses null values for irrelevant fields that still need to be included
  def add_to_classification_store(self, id, text):
    self.classification_store['id'].append(id)
    self.classification_store['tweet_text'].append(text)
    self.classification_store['tweet_label'].append(' ')
    self.classification_store['claim'].append(' ')
    self.classification_store['tweet_date'].append(' ')
    self.classification_store['num_hashtags'].append(0)
    self.classification_store['hashtag_text'].append(' ')
    self.classification_store['has_link'].append(False)
    self.classification_store['polarity'].append(' ')
    self.classification_store['subjective'].append(False)
    self.classification_store['pos_words'].append(0)
    self.classification_store['neg_words'].append(0)
    self.classification_store['misinfo_words'].append(0)
    self.classification_store['debunk_words'].append(0) 
    self.classification_store['misinfo_hashtags'].append(0)
    self.classification_store['debunk_hashtags'].append(0)
    self.classification_store['hyperpartisan'].append(0)
    self.classification_store['followers'].append(100)
    self.classification_store['verified'].append(False)


  # function to check if tweet ID is present in the ES database
  def is_tweet_present(self, tweet_id):
    query_body = { 
      "query": {
        "bool" : {
          "must" : {
            "term": {"entities.Tweet.id" : tweet_id }
          }
        }
      }
    }

    result = es.search(index= self.index, body = query_body, size = 1)

    if len(result['hits']['hits']) > 0:
      return True
    else:
      return False

  def is_tweet_present_range(self, tweet_id):
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

  # function to collect tweet body from db
  def pull_tweet_body(self, tweet_id):
    query_body = { 
      "query": {
        "bool" : {
          "must" : {
            "term": {"entities.Tweet.id" : tweet_id }
          }
        }
      }
    }

    result = es.search(index= self.index, body = query_body, size = 1)
    return result['hits']['hits'][0]

  # main function that fills out quoteG with nodes and edges, depending on 
  def pull_quote_chain(self):

    def get_text(body):
      text_arr = []

      if 'text' in body:
        text_arr.append(body['text'])
      elif 'string' in body:
        text_arr.append(body['string'])
      elif 'entities' in body:
        text_arr.append(body['entities']['text'])
      else:
        try:
          full_body = self.pull_tweet_body(body)
          text = get_text(full_body)
          return text
        except:
          return 'no_text'
      
      for sample in text_arr:
        if sample != '':
          return sample
      
      return 'no_text'

    # Print iterations progress
    # progress bar from https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters
    def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
      """
      Call in a loop to create terminal progress bar
      @params:
          iteration   - Required  : current iteration (Int)
          total       - Required  : total iterations (Int)
          prefix      - Optional  : prefix string (Str)
          suffix      - Optional  : suffix string (Str)
          decimals    - Optional  : positive number of decimals in percent complete (Int)
          length      - Optional  : character length of bar (Int)
          fill        - Optional  : bar fill character (Str)
          printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
      """
      percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
      filledLength = int(length * iteration // total)
      bar = fill * filledLength + '-' * (length - filledLength)
      print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
      # Print New Line on Complete
      if iteration == total: 
          print()


    result = es.search(index= self.index, body = self.quoted_only, size = self.query_size)
    quotes = result['hits']['hits']

    i = 0
    length = len(quotes)
    printProgressBar(0, length, prefix = 'Progress:', suffix = 'Complete', length = 50)

    for quote in quotes:

      i += 1
      #print(f'Number {i} of {length} quotes processed:')
      printProgressBar(i, length, prefix = 'Progress:', suffix = 'Complete', length = 50)
      
      # first get quote body and id for tracking quotes through db
      quote_body = quote['_source']['entities']['Tweet'][0]
      quote_id = quote_body['id_str']

      # then get original body and id
      original_body = quote_body['quoted_status']
      original_id = original_body['id_str']

      # checking original tweet is present in the elasticsearch database before adding connection
      # (limiting range of network)

      if self.is_tweet_present(original_id):
        # getting top hashtag for original tweets
        if 'entities' in original_body:
          if ('hashtags' in original_body['entities']) and (original_body['entities']['hashtags'] != []):
            original_hashtag = original_body['entities']['hashtags'][0]['text']
            if isinstance(original_hashtag, list):
              original_hashtag = original_hashtag[0]
          else:
            original_hashtag = 'None'

        # gexf can't take non string/integer attributes : for now, just get first hashtag
        quote_hashtag = quote['_source']['entities']['Hashtag'][0]['text']

        # get text for use by classifier from quote and classify
        quote_text = get_text(quote_body)
        if quote_text == 'no_text':
          quote_class = 'None'
        else:
          quote_class = Classifier.get_classification_category(quote_text)
          self.add_to_classification_store(quote_id, quote_text)

        # get text for use by classifier from original and classify
        original_text = get_text(original_body)
        if original_text == 'no_text':
          original_class = 'None'
        else:
          original_class = Classifier.get_classification_category(original_text)
          self.add_to_classification_store(original_id, original_text)

        

        # adding nodes into the networkx graph, with hashtags as attributes
        self.quoteG.add_node(quote_id, hashtag = quote_hashtag, text = quote_text, misinfo_class = quote_class)
        self.quoteG.add_node(original_id, hashtag = original_hashtag, text = original_text, misinfo_class = original_class)
        self.quoteG.add_edge(original_id, quote_id)

      # if a further quote object found, continue linking (represented through quoted_status_id_str)
      if 'quoted_status_id_str' in original_body.keys():
        further_id = original_body['quoted_status_id_str']

        if self.is_tweet_present(further_id):
          print('further found within db')
          further_body = self.pull_tweet_body(further_id)

          if 'entities' in further_body:
            if ('hashtags' in further_body['entities']) and (further_body['entities']['hashtags'] != []):
              further_hashtag = further_body['entities']['hashtags'][0]['text']
            else:
              further_hashtag = 'None'
          else:
            further_hashtag = 'None'

          further_text = get_text(further_body)
          if further_text == 'no_text':
            further_class = 'None'
          else:
            further_class = Classifier.get_classification_category(further_text)
            self.add_to_classification_store(further_id, further_text)

          self.quoteG.add_node(further_id, hashtag = further_hashtag, text = further_text, misinfo_class = further_class)
          self.quoteG.add_edge(further_id, original_id)
        else:
          self.quoteG.add_node(further_id, hashtag = 'Outside DB', text = 'Outside DB', misinfo_class = 'Outside DB')
          self.quoteG.add_edge(further_id, original_id)

  def calculate_retweets(self):

    unique_id_store = self.unique_id_store
    start_date = self.start_date
    end_date = self.end_date

    query_body = { 
      "query": {
        "bool" : {
          "must" : {
            "exists": {
                  'field': 'entities.Tweet.retweeted_status'
            }
          },
            "filter": {
              "range": {"entities.Tweet.created_at": {"gte": start_date,"lte": end_date} }
            }
        }
      }
    }

    result = es.search(index = self.index, body = query_body, size = self.query_size)

    retweets = result['hits']['hits']
    for tweet in retweets :
      original_id = tweet['_source']['entities']['Tweet'][0]['retweeted_status']['id_str']
      if original_id in unique_id_store:
        unique_id_store[original_id] += 1
      else:
        unique_id_store[original_id] = 0
    
    nx.set_node_attributes(self.quoteG, unique_id_store, name = 'calculated_retweets')

  def classify_misinfo(self):

    for key in self.classification_store:
      print(f"Len of {key} is {len(self.classification_store[key])}" )

    test_df = pd.DataFrame.from_dict(self.classification_store)
    test_df.to_csv('my_example.csv')

    misinfo_classifier = MisinfoClassifier()
    misinfo_dictionary = misinfo_classifier.run_classifier(test_df)
    print(misinfo_dictionary)

    nx.set_node_attributes(self.quoteG, misinfo_dictionary, name = 'tweet_label')


    



    





      


    




