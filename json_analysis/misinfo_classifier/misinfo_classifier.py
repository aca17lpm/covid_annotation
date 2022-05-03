#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 17:45:59 2022

@author: jacobcrawley

30/04/2022
Used with Jacob Crawley's permission by Lucian Murdin:
slight edits made

"""
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neural_network import MLPClassifier
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.corpus import sentiwordnet as swn


class MisinfoClassifier:

  #placeholder method for class structure
  def __init__(self):
    pass

  ##### ----- Preprocessing methods ----- #####
  def preprocess_tweets(self, data,train_data):
      data = self.edit_tweet_text(data)
      data = self.get_extra_columns(data,train_data)
      data = self.add_top_hashtags(data,train_data)
      return data

  def edit_tweet_text(self, data):

      STOPLIST = ['covid19','covid','coronavirus','https','@','?','-','&amp','will']
      en_stopwords = stopwords.words('english')

      for row in data.itertuples():
          text = self.remove_punctuation(data.at[row.Index,'tweet_text'])
          if self.has_url(text):
              # find and remove urls
              data.at[row.Index,'has_link'] = True
              text = re.sub('http\S+', '',text)
          words = text.split()
          for i in range(len(words)):
              # Lowercase words unless they are all caps or hashtags
              if words[i] != words[i].upper() and words[i][0] != '#':
                  words[i] = words[i].lower()
          words = [w for w in words if w not in en_stopwords]
          text = ' '.join(words)
          data.loc[row.Index,'tweet_text'] = text
      return data

  def get_extra_columns(self, data,train_data):
      """Process tweets to add new layers to data object """
      analyzer = SentimentIntensityAnalyzer()
      misinfo_words = self.get_top_words(train_data,'MISINFO')
      debunk_words = self.get_top_words(train_data,'DEBUNK')
      for row in data.itertuples():
          text = data.at[row.Index,'tweet_text']
          hashtags = self.get_hashtags(text) 
          data.at[row.Index,'hashtag_text'] = hashtags
          if len(hashtags.split()) > 0:
              # find and remove hashtags
              data.at[row.Index,'num_hashtags'] = len(hashtags.split()) 

          words = text.split()

          for i in range(len(words)):
              if words[i] in misinfo_words:
                  data.at[row.Index,'misinfo_words'] += 1
              if words[i] in debunk_words:
                  data.at[row.Index,'debunk_words'] += 1
              try:
                  sent_scores = list(swn.senti_synsets(words[i]))[0]
                  pos = sent_scores.pos_score()
                  neg = sent_scores.neg_score()
                  
                  if pos > 0:
                      data.at[row.Index,'pos_words'] += 1
                  if neg > 0:
                      data.at[row.Index,'neg_words'] += 1
              except:
                  pass         
          data.at[row.Index,'polarity'] = self.get_sentiment(analyzer,text)
          data.at[row.Index,'subjective'] = self.get_subjectivity(data.at[row.Index,'polarity'])
          data.at[row.Index,'tweet_label'] = self.map_label(data.at[row.Index,'tweet_label'])
          # data.at[row.Index,'followers'] = float(data.at[row.Index,'followers'].replace(',',''))
      return data

  def add_top_hashtags(self, data,train_data):
      misinfo_tags = self.get_top_hashtags(train_data, 'MISINFO')
      debunk_tags = self.get_top_hashtags(train_data,'DEBUNK')

      for row in data.itertuples():
          tag_list = data.at[row.Index,'hashtag_text'].split()
          for tag in tag_list:
              if tag.strip() in misinfo_tags:
                  data.at[row.Index,'misinfo_hashtags'] += 1
              if tag.strip() in debunk_tags:   
                  data.at[row.Index,'debunk_hashtags'] += 1
      return data


  def get_sentiment(self, analyzer,text):
      # Returns positive/negative/neutral based on polarity score
      scores = analyzer.polarity_scores(text)
      polarity = scores['compound']
      
      if polarity > 0:
          return 'positive'
      if polarity < 0:
          return 'negative'
      return 'neutral'
      
  def get_subjectivity(self, polarity):
      if polarity == "neutral":
          return False
      return True
    
  def remove_punctuation(self, text):
      punct = '[^\w\s#@]'
      return re.sub(punct,'',text)
          
  def has_url(self, text):
      urls = re.findall('http\S+', text)
      if len(urls) > 0:
          return True
      return False

  def get_hashtags(self, text):
      return ' '.join(re.findall("#(\w+)", text))

  def map_label(self, label):
      # Convert all misinfo/debunk types into the same label
      if "MISINFO" in label:
          return "MISINFO"
      elif "DEBUNK" in label:
          return "DEBUNK"
      else:
          return label

  def plot_confusion_matrix(labels,lab_set,pred):
      sn.heatmap(confusion_matrix(labels,pred), annot=True, fmt='g',xticklabels=lab_set,yticklabels=lab_set)
      plt.xlabel('Predicted label')
      plt.ylabel('Actual label')
      plt.show()

  def get_label_freq(labels):
      label_freq = dict()
      for l in labels:
          if l not in label_freq:
              label_freq[l] = 1
          else:
              label_freq[l] += 1
      return label_freq

  def get_top_words(self, tweets,label):
      # Return the x most frequent words from a set of tweets for a given label
      label_words = self.get_tweet_text(tweets, label) # dictionary of {word: frequency} pairs from all the tweets
      top_words = sorted(label_words, key=label_words.get, reverse=True)[:1000]
      return top_words

  def get_top_hashtags(self, tweets, label):
      tags = dict()
      for row in tweets.itertuples():
          if tweets.at[row.Index,'tweet_label'] == label:
              if isinstance(tweets.at[row.Index,'hashtag_text'],str):
                  tag_list = tweets.at[row.Index,'hashtag_text'].split()
                  for tag in tag_list:
                      if tag.strip() in tags:
                          tags[tag] += 1
                      else:
                          tags[tag] = 1
      return sorted(tags, key=tags.get, reverse=True)[:200]

  def get_tweet_text(self, tweetset,label):
      # Returns list of {word: frequency} pairs from the tweets
      tweets = ""
      for row in tweetset.itertuples():
          if tweetset.at[row.Index,'tweet_label'] == label:
              tweets += tweetset.at[row.Index,'tweet_text']
          
      filter_tweets = self.get_tweet_list(tweets)
      tweet_dict = dict()
      for tweet in filter_tweets:
          if tweet not in tweet_dict:
              tweet_dict[tweet] = 1
          else:
              tweet_dict[tweet] += 1
      return tweet_dict

  def get_tweet_list(self, tweets):
      # From string, create list of words and remove ones with # at start
      tweet_list = tweets.split()
      filtered = list()

      STOPLIST = ['covid19','covid','coronavirus','https','@','?','-','&amp','will']
      en_stopwords = stopwords.words('english')
      
      for t in tweet_list:
          #if t[0] != '#' and t.lower() not in STOPLIST:
          #   filtered.append(t)
              
          valid = True
          if t[0] == '#':
              valid = False
          if t.lower() in en_stopwords:
              valid = False
          for word in STOPLIST:
              if word in t.lower():
                  valid = False 
          if valid:
              filtered.append(t)
      return filtered

  def get_word_embeddings(text_data,embedding_model):
      # Create data frame of word2vec word embedding for each tweet 
      vectorizer = CountVectorizer(stop_words='english')
      text_vector = vectorizer.fit_transform(text_data).toarray()
      w2vec_data = pd.DataFrame()

      for i in range(len(text_vector)):
          sentence = np.zeros(200)
          words = text_data[i].split()
          for w in words:
              # if w in embedding_model.key_to_index.keys():
              #     sentence += GoogleModel[w]
              if w in embedding_model:
                  sentence += embedding_model[w]
          w2vec_data = w2vec_data.append(pd.DataFrame([sentence]))
      return w2vec_data      

  def load_glove():
      filename = "./glove.twitter.27B/glove.twitter.27B.200d.txt"
      with open(filename, encoding="utf8") as f:
          content = f.readlines()
      model = {}
      for line in content:
          line_split = line.split()
          word = line_split[0]
          embedding = np.array([float(val) for val in line_split[1:]])
          model[word] = embedding
      return model

  def get_tfidf_vectors(self, text_data,polarity_data):
      text_vectorizer = TfidfVectorizer(max_features=200,min_df=3, max_df=0.7, stop_words=stopwords.words('english'))
      text_vector = text_vectorizer.fit_transform(text_data).toarray()

      sent_vect = TfidfVectorizer() # vectorise pos/neutral/negative sentiment features
      sent_vector = sent_vect.fit_transform(polarity_data).toarray()
      return text_vector, sent_vector
      
  def get_bert(text_data):
      model_class, tokenizer_class, pretrained_weights = (transformers.DistilBertModel, transformers.DistilBertTokenizer, 'distilbert-base-uncased')
      # Load pretrained model/tokenizer
      tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
      model = model_class.from_pretrained(pretrained_weights)
      tokenized = text_data.swifter.apply(lambda x: tokenizer.encode(x, add_special_tokens=True))
      max_len = 0
      for i in tokenized.values:
          if len(i) > max_len:
              max_len = len(i)
      padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])

      attention_mask = np.where(padded != 0, 1, 0)
      input_ids = torch.tensor(padded)
      attention_mask = torch.tensor(attention_mask)
      with torch.no_grad(): last_hidden_states = model(input_ids, attention_mask=attention_mask)
      features = last_hidden_states[0][:, 0, :].numpy()
      return features
      
  def get_csv_words(filename):
      data = pd.read_csv(filename)
      return data['word'].values

  def run_classifier(self, test_df):
      train_data = pd.read_csv('C:\codeyr3\dissertation\covid_annotation\json_analysis\misinfo_classifier\elastic-tweets-train-processed.csv')
      feats = train_data[['num_hashtags','has_link','subjective','pos_words','neg_words','misinfo_words','debunk_words','misinfo_hashtags','debunk_hashtags']]
      labels = train_data['tweet_label']
      tweets = train_data['tweet_text']
      sentiment = train_data['polarity']
      text_vector, sent_vector = self.get_tfidf_vectors(tweets,sentiment)
      features = np.concatenate((text_vector,sent_vector,feats.to_numpy()),axis=1)
      tweets = train_data['tweet_text']
      
      #test_data = pd.read_csv('C:\codeyr3\dissertation\covid_annotation\json_analysis\my_example.csv')
      test_data = test_df
      test_data = self.preprocess_tweets(test_data,train_data)

      print(f"Test shape is :{test_data.shape}")
      print(f"Training shape is :{train_data.shape}")
      print(f"Test columns are :{list(test_data.columns)}")
      print(f"Training columns  :{list(train_data.columns)}")

      train_data = features
      train_labels = labels

      print(f"Features of training data: {train_data}")

      ## Prepare test set for classification ###
      test_tweets = test_data['tweet_text']
      test_sentiment = test_data['polarity']
      test_feats = test_data[['num_hashtags','has_link','subjective','pos_words','neg_words','misinfo_words','debunk_words','misinfo_hashtags','debunk_hashtags']]
      test_text, test_sent = self.get_tfidf_vectors(test_tweets,test_sentiment)
      test_features = np.concatenate((test_text,test_sent,test_feats.to_numpy()),axis=1)

      print(f"Test features shape is :{test_features.shape}")
      print(f"Training features shape is :{features.shape}")
    
      #### Unlabelled set classification ####
      model = MLPClassifier(max_iter=300)
      model.fit(train_data,train_labels)
      pred = model.predict(test_features)

      for i in range(len(test_data)):
          test_data.at[i,'tweet_label'] = pred[i] 

      #test_data.to_csv('./elastic-test-classified.csv') # will make a new CSV file - rename with whatever you want the output to be 

      keys = test_data['id'].to_list()
      keys = list(map(str, keys))
      
      print(type(keys[0]))

      values = test_data['tweet_label'].to_list()
      values = list(map(str, values))
      print(type(values[0]))

      zip_iterator = zip(keys, values)
      misinfo_dictionary = dict(zip_iterator)

      return misinfo_dictionary

    #def train_classifier():
      



##### ----- Main Program ----- #####
#run_classifier('elastic-sample.csv')



    