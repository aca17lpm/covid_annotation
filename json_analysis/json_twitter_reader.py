import json

with open('c:/codeyr3/dissertation/covid_annotation/json_dumps/misinfo_debunk_user_init.json', encoding="utf-8") as f:
    data = json.load(f)

print(str(data[0]))