#url = https://cloud-api.gate.ac.uk/process/covid19-misinfo

import json
import asyncio
import requests
import os

class Classifier:

    def get_classification_category(text):
        class_token = os.environ.get('CLASSIFIER_TOKEN')
        auth_token = class_token.split(':')

        key = auth_token[0]
        password = auth_token[1]

        url = 'https://cloud-api.gate.ac.uk/process/rumour-veracity'
        data = {"text": text}

        response = requests.post(url, json=data, auth=(key,password), timeout=1280)
        print(response)

        if response.status_code == 200:
            print(response.json()['entities']['Veracity'][0]['rumour_label'])
            return response.json()['entities']['Veracity'][0]['rumour_label']
        else:
            return 0

    # for i in get_request('http://gateservice10.dcs.shef.ac.uk:9300/_cat/indices?format=json&pretty'):
    #   print(i['index'])

print(Classifier.get_classification_category("Definitely not 3000 people (the official death toll), adding another 0 or two 0's is also possible."))