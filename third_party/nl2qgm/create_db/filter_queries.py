import os
import json
from ratsql.datasets.spider_lib.process_sql import Schema, get_schema, get_sql

target_dbs = ['poker_player', 'singer', 'orchestra', 'concert_singer', 'network_1', 'wta_1', 'car_1', 'world_1', 'real_estate_properties']

top3_dbs = ['poker_player', 'singer', 'orchestra']
middle3_dbs = ['concert_singer', 'network_1', 'wta_1']
bottom3_dbs = ['car_1', 'world_1', 'real_estate_properties']

table_file = '/root/NL2QGM/data/spider_modified/tables_modified.json'
original_file = '/root/NL2QGM/data/spider/dev.json'

if __name__ == "__main__":
    db_dir = '/root/NL2QGM/data/spider_modified/database/'
    dbs = {item['db_id']: item for item in json.load(open(table_file))}
    db_names = dbs.keys()
    schemas = {}
    for db_name in db_names:
        db_path = os.path.join(db_dir, db_name, db_name + '.sqlite')
        schemas[db_name] = Schema(get_schema(db_path), dbs[db_name])

    data = json.load(open(original_file))
    target_items = []
    for datum in data:
        db_id = datum['db_id']
        if datum['db_id'] in target_dbs:
            print(f"{datum['query']}")
            if db_id in top3_dbs:
                datum['db_id'] = 'top3_dbs'
            elif db_id in middle3_dbs:
                datum['db_id'] = 'middle3_dbs'
            elif db_id in bottom3_dbs:
                datum['db_id'] = 'bottom3_dbs'
            else:
                raise RuntimeError("Something Wrong")

            # Re-parse SQL 
            parsed_sql = get_sql(schemas[datum['db_id']], datum['query'])
            datum['sql'] = parsed_sql
            target_items.append(datum)

    target_file = '/root/NL2QGM/data/spider_modified/filtered_dev.json'
    print(f"Writing... len:{len(target_items)}")
    with open(target_file, 'w') as f:
        f.write(json.dumps(target_items, indent=4))
    print("All Done!")
