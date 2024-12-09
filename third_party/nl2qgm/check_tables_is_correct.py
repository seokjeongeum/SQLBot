import json

file_path = "/root/NL2QGM/data/spider_modified/tables_modified.json"
dbs = json.load(open(file_path))
dbs = {item['db_id']:item for item in dbs}

db_ids = ["top3_dbs", "middle3_dbs", "bottom3_dbs"]

for db_id in db_ids:
    db = dbs[db_id]
    assert len(db['column_names']) == len(db['column_names_original'])
    assert len(db['column_names']) == len(db['column_types'])
    assert len(db['table_names']) == len(db['table_names_original'])

print("Good!")