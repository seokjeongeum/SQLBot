import json

file_path = "/root/NL2QGM/data/spider/dev.json"

data = json.load(open(file_path))

cnt = 0
for idx, item in enumerate(data):
    tables = item['sql']['from']['table_units']
    if tables[0][0] == 'sql':
        continue
    tables = list(map(lambda k: tuple(k), tables))

    if len(tables) != len(set(tables)):
        cnt += 1
        print(f"\nidx:{idx}")
        print(f"tables:{tables}")
        print(f"query:{item['query']}")

print(f"Total cnt:{cnt}")
