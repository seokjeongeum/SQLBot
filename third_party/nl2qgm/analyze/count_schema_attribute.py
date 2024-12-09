import json
file_path = "/root/NL2QGM/data/spider/tables.json"

data = json.load(open(file_path))

def avg(items):
    return sum(items) / len(items)

# Get information
dbs = []
for item in data:
    db = {'db_id': item['db_id'], 'table_cnt': len(item['table_names']), 'column_cnt': len(item['column_names'])-1}
    dbs.append(db)


table_lens = list(map(lambda k: k['table_cnt'], dbs))
table_lens = sorted(table_lens)
column_lens = list(map(lambda k: k['column_cnt'], dbs))
column_lens = sorted(column_lens)

print(f"Average table length:{avg(table_lens)} Average column length:{avg(column_lens)}")

# 
stop = 1
