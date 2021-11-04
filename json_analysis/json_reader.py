import json
import matplotlib.pyplot as plt

with open('c:/codeyr3/dissertation/wv_covid_annotation/mergedData/merged_clean.json', encoding="utf-8") as f:
    data = json.load(f)


print(len(data))

# Start with number of annotations per class, sort through dictionary to query each annotated claim, find its class, then represent with a bar chart

print(data[0].get('annotations')[0].get('label'))

class_labels = []

for annotation in data:
    class_label = annotation.get('annotations')[0].get('label')
    class_labels.append(class_label)



class_label_count = {
    'VirTrans' : 0,
    'VirOrgn' : 0,
    'GenMedAdv' : 0,
    'CommSpread' : 0,
    'Consp' : 0,
    'PromActs' : 0,
    'PubAuthAction' : 0,
    'PubPrep' : 0,
    'Vacc' : 0,
    'Prot' : 0,
    'None' : 0
}


for class_label in class_labels:
    class_label_count[class_label] += 1

classes = list(class_label_count.keys())
counts = list(class_label_count.values())

plt.bar(classes, counts)
plt.title('COVID claims categorised into classes')
plt.xlabel('Classes')
plt.ylabel('Number of claims')
plt.show()
