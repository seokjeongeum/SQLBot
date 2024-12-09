import os
import copy
import json
from ratsql.datasets.spider_lib.process_sql import Schema, get_schema, get_sql

target_dbs = ['poker_player', 'singer', 'orchestra', 'concert_singer', 'network_1', 'wta_1', 'car_1', 'world_1', 'real_estate_properties']

data_dir = "/root/NL2QGM/data/spider/"
db_dir = os.path.join(data_dir, "database")
table_file_path = os.path.join(data_dir, "tables_extended.json")
dev_file_path = os.path.join(data_dir, "dev.json")
new_dev_file_path = os.path.join(data_dir, "dev_extended.json")

if __name__ == "__main__":
    db_data = json.load(open(table_file_path))
    db_data = {item['db_id']: item for item in db_data}
    dev_data = json.load(open(dev_file_path))
    print(f"Original Dev set: {len(dev_data)}")

    # Filter datas
    dev_data = list(filter(lambda k: k['db_id'] in target_dbs, dev_data))
    print(f"Filtered Dev set: {len(dev_data)}")

    # Group queries by DB
    queries_by_db = {}
    for item in dev_data:
        db_id = item['db_id']
        if db_id not in queries_by_db.keys():
            queries_by_db[db_id] = [item]
        else:
            queries_by_db[db_id] += [item]

    # Create Schema object
    schemas = {}
    for db_name in db_data.keys():
        db_path = os.path.join(db_dir, db_name, db_name + '.sqlite')
        schemas[db_name] = Schema(get_schema(db_path), db_data[db_name])

    # Create new query items for each of new DBs
    new_data_list = []
    db_data = dict(filter(lambda k: '__' in k[0], db_data.items()))
    for new_db_id, db in db_data.items():
        base_db_name = new_db_id.split('__')[0]
        assert base_db_name == db['db_id_list'][0]
        assert base_db_name in queries_by_db.keys()
        target_group_queries = queries_by_db[base_db_name]
        for item in target_group_queries:
            new_item = copy.deepcopy(item)
            new_item['db_id'] = new_db_id
            # Re-parse SQL
            new_item['sql'] = get_sql(schemas[new_db_id], new_item['query'])
            new_data_list.append(new_item)

    print(f"New Dev data len:{len(new_data_list)}")

    # Save
    print(f"Writing file:{new_dev_file_path}...")
    with open(new_dev_file_path, "w") as f:
        f.write(json.dumps(new_data_list, indent=4))

    print("Done!")
