import json

with open("data/spider/dev.json") as f:
    dev = json.load(f)

with open("data/spider/qgm_dev_parsable.json") as f:
    qgm_dev = json.load(f)

qgm_dev_dict = dict()

for e in qgm_dev:
    start_idx = e['actions'].index("SUPER_COL_EXIST_NO")
    qgm_dev_dict[e["db_id"] + e['question']] = {
        'states': e['states'][start_idx + 1:],
        'actions': e['actions'][start_idx + 1:]
    }

new_dev = []
for e in dev:

    if e["db_id"] + e["question"] in qgm_dev_dict:
        e['states'] = qgm_dev_dict[e["db_id"] + e["question"]]['states']
        e['actions'] = qgm_dev_dict[e["db_id"] + e["question"]]['actions']
        new_dev.append(e)
print(f'{len(new_dev)}, {len(qgm_dev)}')


with open("data/spider/dev_qgm.json", "w") as f:
    f.write(json.dumps(new_dev))

with open("data/spider/train.json") as f:
    train = json.load(f)

with open("data/spider/qgm_train_parsable.json") as f:
    qgm_train = json.load(f)

qgm_train_dict = dict()

for e in qgm_train:
    start_idx = e['actions'].index("SUPER_COL_EXIST_NO")
    qgm_train_dict[e["db_id"] + e['question']] = {
        'states': e['states'][start_idx + 1:],
        'actions': e['actions'][start_idx + 1:]
    }

new_train = []
for e in train:
    if e["db_id"] + e["question"] in qgm_train_dict:
        e['states'] = qgm_train_dict[e["db_id"] +e["question"]]['states']
        e['actions'] = qgm_train_dict[e["db_id"] +e["question"]]['actions']
        new_train.append(e)

with open("data/spider/train_qgm.json", "w") as f:
    f.write(json.dumps(new_train))

print(f'{len(new_train)}, {len(qgm_train)}')

