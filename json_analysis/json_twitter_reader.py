import json
import pandas as pd
import requests
import os

with open('c:/codeyr3/dissertation/covid_annotation/json_dumps/misinfo_debunk_user_init.json', encoding="utf-8") as f:
    data = json.load(f)

# for line in data:
#   if line['tweet_label'] == 'MISINFO': print(str(line))

#print(len(data))

# need to form twitter authorization headers
def connect_to_twitter():
    # secure version to get bearer token from environment variable
    bearer_token = os.environ.get("BEARER_TOKEN")
    return {"Authorization": "Bearer {}".format(bearer_token)}
headers = connect_to_twitter()

def make_request(headers):
    url = "https://api.twitter.com/2/tweets/1491217291669049344"
    return requests.request("GET", url, headers=headers).json()
response = make_request(headers)
print(response)

# For each tweet in dataset, find number of retweets associated