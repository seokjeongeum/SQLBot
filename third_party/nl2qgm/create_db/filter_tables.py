import os 
import json

file_path = "/root/NL2QGM/data/spider_modified/tables_modified.json"

data = json.load(open(file_path))
target_dbs = ['top3_dbs', 'middle3_dbs', 'bottom3_dbs']
data = list(filter(lambda k: k['db_id'] in target_dbs, data))

print(f"Writing...! Total:{len(data)}")
with open(file_path, 'w') as f:
    f.write(json.dumps(data, indent=4))
print("Done!")
