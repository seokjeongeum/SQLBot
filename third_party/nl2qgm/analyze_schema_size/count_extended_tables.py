import os
import json

original_file_path = "/root/NL2QGM/data/spider/tables.json"
extended_file_path = "/root/NL2QGM/data/spider/tables_extended.json"


def find_db_info_by_db_name(dbs, db_name):
    for item in dbs:
        if item['db_id'] == db_name:
            return item
    raise RuntimeError(f"Bad db_name:{db_name}")


if __name__ == "__main__":
    # Load 
    ori_data = json.load(open(original_file_path))
    extended_data = json.load(open(extended_file_path))

    # Find the base dbs
    filtered_data = list(filter(lambda k: '__' in k['db_id'], extended_data))

    assert len(filtered_data) == len(extended_data)
    base_dbs = list(set(map(lambda k: k['db_id'].split('__')[0], filtered_data)))

    # Create DB mapping
    db_mapping = {}
    for base_db in base_dbs:
        db_mapping[base_db] = []

        for item in filtered_data:
            if base_db in item['db_id']:
                db_mapping[base_db] += [item['db_id']]

    # Show 
    for base_db_name, item_list in db_mapping.items():
        original_db = find_db_info_by_db_name(ori_data, base_db_name)
        ori_tab_num = len(original_db['table_names'])
        ori_col_num = len(original_db['column_names'])
        print(f"Original DB:{base_db_name} tab:{ori_tab_num} col:{ori_col_num}")
        for extended_db_name in item_list:
            extended_db = find_db_info_by_db_name(filtered_data, extended_db_name)
            tab_num = len(extended_db['table_names']) - ori_tab_num
            col_num = len(extended_db['column_names']) - ori_col_num
            print(f"\tdb_name:{extended_db_name} tab:{tab_num} col:{col_num}")

    print("Done!")