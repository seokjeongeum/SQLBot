import os
import json
import copy
import random
import functools


dev_file_path = "/root/NL2QGM/data/spider/dev.json"
original_file_path = "/root/NL2QGM/data/spider/tables.json"
new_file_path = original_file_path.replace("tables.json", "tables_extended.json")

# DBs that are going to be extended
grouped_dbs = ["poker_player", "singer", "orchestra", "concert_singer", "network_1", "wta_1", "car_1", "world_1", "real_estate_properties"]


def assert_correct_length(data):
    assert len(data['table_names']) == len(data['table_names_original'])
    assert len(data['column_names']) == len(data['column_names_original'])
    assert len(data['column_names']) == len(data['column_types'])
    for item in data['column_names']:
        assert item[0] < len(data['table_names'])
    for item in data['column_names_original']:
        assert item[0] < len(data['table_names'])
    for item in data['primary_keys']:
        assert item < len(data['column_names'])
    for item in data['foreign_keys']:
        assert item[0] < len(data['column_names']) and item[1] < len(data['column_names'])
    return True


def add_number(item, base_idx):
    if type(item) == list:
        assert len(item) == 2
        new_item = [add_number(item[0], base_idx), add_number(item[1], base_idx)]
    elif type(item) == int:
        new_item = item + base_idx
    elif type(item) == str:
        new_item = item
    else:
        raise RuntimeError(f"Unexpected type:{type(item)} ({item})")
    return new_item


def change_base_idx(ori_list, base_idx):
    new_list = []
    for item in ori_list:
        new_list.append(add_number(item, base_idx))
    return new_list

def append_db(des_data, src_data):
    # This order of keys is important for correct indexing
    keys = ['primary_keys', 'foreign_keys', 'column_names', 'column_names_original',
            'table_names', 'table_names_original', 'column_types']
    
    # Append Datas
    for key in keys:
        # Find new idx for table and column
        if des_data[key]:
            if key in ['column_names', 'column_names_original']:
                base_idx = len(des_data['table_names'])
            else:
                base_idx = len(des_data['column_names']) - 1
        else:
            base_idx = 0
        
        # Remove * if already exist
        if base_idx > 0 and key in ['column_types', 'column_names', 'column_names_original']:
            new_data = src_data[key][1:]
        else:
            new_data = src_data[key]

        des_data[key] += change_base_idx(new_data, base_idx)

    return des_data


def append_dbs(ori_data):
    new_dbs = []
    candidate_keys = list(filter(lambda k: k not in grouped_dbs, ori_data.keys()))
    random.shuffle(candidate_keys)

    print(f"Destin DBs:{len(grouped_dbs)} src DBs:{len(candidate_keys)}")

    # For all grouped DBs
    for dest_db_id in grouped_dbs:
        for source_len in range(len(candidate_keys)):
            # Get target DBs
            current_source_keys = candidate_keys[:source_len+1]
            candidate_dbs = [item for key, item in ori_data.items() if key in current_source_keys]
            # Create New DB
            new_db = copy.deepcopy(ori_data[dest_db_id])
            new_db.update({"db_id": f"{dest_db_id}__{source_len+1}", "db_id_list": [dest_db_id] + current_source_keys})
            new_db = functools.reduce(append_db, candidate_dbs, new_db)
            new_dbs.append(new_db)

    return new_dbs


if __name__ == "__main__":
    # Random seed
    random.seed(0)

    # Load
    data = json.load(open(original_file_path))
    data = {item['db_id']: item for item in data}
    print(f"Total DBs in tables.json:{len(data)}")
    # Filter by tables in dev. set
    dev_data = set(map(lambda k: k['db_id'], json.load(open(dev_file_path))))
    data = dict(filter(lambda k: k[0] in dev_data, data.items()))
    print(f"Total DBs in dev set:{len(data)}")

    # assert grouped_dbs do exist
    flags = list(map(lambda k: k in data.keys(), grouped_dbs))
    assert False not in flags, f"item in grouped_dbs does not exist in the {original_file_path}"

    # Append 
    new_data = append_dbs(data)
    
    # Check no weird numbers
    list(map(assert_correct_length, new_data))

    # Write
    with open(new_file_path, "w") as f:
        f.write(json.dumps(new_data, indent=4))
