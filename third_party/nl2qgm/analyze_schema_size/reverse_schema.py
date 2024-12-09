import os
import json
from ratsql.datasets.spider_lib.process_sql import Schema, get_schema, get_sql

table_file_path = "/root/NL2QGM/data/spider/tables_extended.json"
dev_file_path = "/root/NL2QGM/data/spider/dev_extended.json"
db_dir = "/root/NL2QGM/data/spider/database/"


def reverse_column(item):
    return [item[0]] + list(reversed(item[1:]))


def modify_tab_reference(items, tab_len):
    for item in items:
        if item[0] == -1:
            continue
        item[0] = tab_len-1 - item[0]
    return items


def modify_reference_for_keys(items, col_len):
    new_items = []
    for item in items:
        if type(item) == list:
            assert len(item) == 2
            item[0] = col_len - item[0]
            item[1] = col_len - item[1]
        else:
            item = col_len - item
        new_items.append(item)
    return new_items


def modify_tables_json(dbs):
    for item in dbs:
        # Reverse tables
        keys = ['table_names', 'table_names_original']
        for key in keys:
            item[key] = list(reversed(item[key]))

        # Reverse columns
        keys = ['column_names', 'column_names_original']
        for key in keys:
            item[key] = reverse_column(item[key])
        item['column_types'] = list(reversed(item['column_types']))
        
        # fix table reference in columns
        for key in keys:
            item[key] = modify_tab_reference(item[key], len(item['table_names']))
        
        # modify foreign/primary keys
        keys = ['foreign_keys', 'primary_keys']
        for key in keys:
            item[key] = modify_reference_for_keys(item[key], len(item['column_names']))
    return dbs


def modify_dev_json(dev_items, dbs):
    db_data = {item['db_id']: item for item in dbs}

    # Create Schema object
    schemas = {}
    for db_name in db_data.keys():
        db_path = os.path.join(db_dir, db_name, db_name + '.sqlite')
        schemas[db_name] = Schema(get_schema(db_path), db_data[db_name])

    # Create correct 'sql' for the reversed table
    for item in dev_items:
        item['sql'] = get_sql(schemas[item['db_id']], item['query'])

    return dev_items


if __name__ == "__main__":
    dev_items = json.load(open(dev_file_path))
    dbs = json.load(open(table_file_path))

    dbs = modify_tables_json(dbs)
    dev_items = modify_dev_json(dev_items, dbs)

    # Write file
    print("Writing...")
    with open(dev_file_path.replace(".json", "_reversed.json"), "w") as f:
        f.write(json.dumps(dev_items, indent=4))
    with open(table_file_path.replace(".json", "_reversed.json"), "w") as f:
        f.write(json.dumps(dbs, indent=4))
    print("Done!")
