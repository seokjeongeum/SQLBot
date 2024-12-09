import json


file_path = "/root/NL2QGM/data/spider/train_others.json"
table_path = "/root/NL2QGM/data/spider/tables.json"

tables = json.load(open(table_path))
db = {}
for table in tables:
    db[table['db_id']] = table

data = json.load(open(file_path))

cnt = 0
db_ids = []
for datum in data:
    db_id = datum['db_id']
    conds = datum['sql']['from']['conds']
    if not conds:
        continue

    conds = [item for idx, item in enumerate(conds) if idx % 2 == 0]
    for cond in conds:
        # check if PK relation
        table_1 = cond[2][1][1]
        table_2 = cond[3][1]
        pk_keys = db[db_id]['foreign_keys']
        if [table_1, table_2] in pk_keys or [table_2, table_1] in pk_keys:
            # Is pk relation
            stop = 1
        else:
            # Not pk relation
            cnt += 1
            db_ids.append(db_id)

print(f"Total:{cnt}/{len(data)} DB_list:{set(db_ids)}")