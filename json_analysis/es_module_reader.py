import os
import pandas as pd

from elasticsearch import ElasticSearch

es = Elasticsearch(
    http_auth=(“username”, “password”)
)