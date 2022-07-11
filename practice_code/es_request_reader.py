import json
import pandas as pd
import requests
import os

def create_elastic_header():
  auth_token = os.environ.get('ELASTIC_TOKEN')
  return {"Authorization": "Basic {}".format(auth_token)}

header = create_elastic_header()

def get_request(url):
  return requests.request("GET", url, headers=header).json()

for i in get_request('http://gateservice10.dcs.shef.ac.uk:9300/_cat/indices?format=json&pretty'):
  print(i['index'])

print(get_request('http://gateservice10.dcs.shef.ac.uk:9300/_cat/indices'))

