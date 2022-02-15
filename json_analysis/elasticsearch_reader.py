import json
import pandas as pd
import requests
import os

def create_elastic_header():
  auth_token = os.environ.get('ELASTIC_TOKEN')
  return {"Authorization": "Basic {}".format(auth_token)}

header = create_elastic_header()

def get_request(header):
  url = "http://gateservice10.dcs.shef.ac.uk:9300/_cat/indices"
  return requests.request("GET", url, headers=header).json()

print(get_request(header))