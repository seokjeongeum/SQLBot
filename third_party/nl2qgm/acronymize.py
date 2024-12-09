from collections import defaultdict
import copy
from sql_metadata import Parser
import shutil
import sqlite3
import argparse
from pathlib import Path
import json
from tqdm import tqdm


def _acronymize_tables(src_dataset_root:Path, dst_dataset_root:Path):
    """acronymize column names in table.json file
    """

    def _acronymize_colname(column_name_original):
        colname_splitted = column_name_original.lower().split('_')
        if len(colname_splitted) > 1:
            colname_acron = '_'.join(map(lambda x: x[0], colname_splitted))
        else:
            colname_acron = colname_splitted[0]
        
        return colname_acron


    # load tables.json file
    with open(src_dataset_root / "tables.json", 'r') as f:
        orig_dbs = json.load(f)
    
    # acronymize column names
    acron_dbs = []
    for orig_db in orig_dbs:
        column_names_original = orig_db["column_names_original"]
        _column_names_acron_original = list(map(lambda x: [x[0], _acronymize_colname(x[1])], column_names_original))

        colname_acron_set = set()
        colname_duplicate_counter = defaultdict(int)
        column_names_acron_original = []
        prev_tid = -1
        for i, (tid, _colname_acron) in enumerate(_column_names_acron_original):
            if prev_tid != tid:
                colname_acron_set = set()
                colname_duplicate_counter = defaultdict(int)
                prev_tid = tid
            if _colname_acron in colname_acron_set:
                colname_duplicate_counter[_colname_acron] = colname_duplicate_counter[_colname_acron] + 1
                colname_acron = f'{_colname_acron}_{colname_duplicate_counter[_colname_acron]}'
            else:
                colname_acron = _colname_acron
            colname_acron_set.add(colname_acron)
            column_names_acron_original.append([tid, colname_acron])
        
        column_names_acron = list(map(lambda x: [x[0], x[1].replace('_', ' ')], column_names_acron_original))

        # create column descriptions
        col_descs = []
        for tid, col_name in orig_db["column_names"]:
            col_desc = f'{col_name} of {orig_db["table_names"][tid]}'
            col_descs.append(col_desc)

        acron_db = copy.deepcopy(orig_db)
        acron_db["column_names_original"] = column_names_acron_original
        acron_db["column_names"] = column_names_acron
        acron_db["column_descriptions"] = col_descs
        acron_dbs.append(acron_db)

    # create acronymized tables.json file
    dst_dataset_root.mkdir(exist_ok=True, parents=True)
    with open(dst_dataset_root / "tables.json", 'w') as f:
        json.dump(acron_dbs, f, indent=2, ensure_ascii=False)

    # create dictionary
    acron_dict = defaultdict(lambda: defaultdict(dict))
    for orig_db, acron_db in zip(orig_dbs, acron_dbs):
        assert orig_db["db_id"] == acron_db["db_id"]
        db_id = orig_db['db_id']
        table_names = orig_db['table_names_original']
        orig_colnames = orig_db["column_names_original"][1:]
        acron_colnames = acron_db["column_names_original"][1:]

        for (tid, orig_colname), (_, acron_colname) in zip(orig_colnames, acron_colnames):
            acron_dict[db_id][table_names[tid]][orig_colname] = acron_colname
            acron_dict[db_id][table_names[tid]][orig_colname.upper()] = acron_colname
            acron_dict[db_id][table_names[tid]][orig_colname.lower()] = acron_colname
            acron_dict[db_id][table_names[tid].upper()][orig_colname] = acron_colname
            acron_dict[db_id][table_names[tid].upper()][orig_colname.upper()] = acron_colname
            acron_dict[db_id][table_names[tid].upper()][orig_colname.lower()] = acron_colname
            acron_dict[db_id][table_names[tid].lower()][orig_colname] = acron_colname
            acron_dict[db_id][table_names[tid].lower()][orig_colname.upper()] = acron_colname
            acron_dict[db_id][table_names[tid].lower()][orig_colname.lower()] = acron_colname


    return acron_dict

def _acronymize_database(src_database_path:Path, dst_database_path:Path, acron_dict):
    """acronymize column names in database sqlite file
    """
    # duplicate src_database to dst_database
    dst_database_path.parent.mkdir(exist_ok=True, parents=True)
    shutil.copyfile(src_database_path, dst_database_path)

    # connect dst_databse
    with sqlite3.connect(dst_database_path) as con:
        cursor = con.cursor()

        # rename columns
        for table_name in tqdm(acron_dict, leave=False, desc=src_database_path.stem):
            if table_name == 'sqlite_sequence': continue
            for orig_col_name in acron_dict[table_name]:
                acron_col_name = acron_dict[table_name][orig_col_name]
                script = f"ALTER TABLE '{table_name}' RENAME COLUMN '{orig_col_name}' TO '{acron_col_name}'" # it may not work before version 3.25.0 sqlite
                cursor.execute(script)

def _acronymize_examples(src_examples_path, dst_examples_path, acron_dict):
    """acronymize column names in database sqlite file
    """
    # load examples file
    with open(src_examples_path, 'r') as f:
        examples = json.load(f)

    # change column names in sql queries
    acron_examples = []
    for example in tqdm(examples, leave=False):
        db_id = example['db_id']
        query = example["query"]
        query_toks = example["query_toks"]
        query_toks_no_value = example["query_toks_no_value"]
        parser = Parser(query)
        cols = parser.columns
        tables = parser.tables

        acron_query = copy.deepcopy(query)
        for col in cols:
            _col = col.split('.')[-1]
            acron_col_candidates = [acron_dict[db_id][table][_col] for table in tables]
            assert len(acron_col_candidates) == 1
            acron_query = acron_query.replace(col, acron_col_candidates[0])
            acron_query_toks = map(lambda x: x.replace(col, acron_col_candidates[0]), query_toks)
            acron_query_toks_no_value = map(lambda x: x.replace(col, acron_col_candidates[0]), query_toks_no_value)
        
        acron_example = copy.deepcopy(example)
        acron_example['db_id'] = db_id
        acron_example['query'] = acron_query
        acron_example['query_toks'] = acron_query_toks
        acron_example['query_toks_no_value'] = acron_query_toks_no_value
        acron_examples.append(acron_example)

    # create acronymized examples file
    with open(dst_examples_path, 'w') as f:
        json.dump(acron_examples, f, indent=2, ensure_ascii=False)


def acronymize(src_dataset_root:Path, dst_dataset_root:Path):
    print("acroynmizing tables.json ...")
    acron_dict = _acronymize_tables(src_dataset_root, dst_dataset_root)

    # print("acroynmizing databases ...")
    # src_database_root = src_dataset_root / "database"
    # for src_database_file in tqdm(list(src_database_root.glob('**/*.sqlite'))):
    #     dst_database_file = dst_dataset_root / f"database/{src_database_file.stem}/{src_database_file.stem}.sqlite"
    #     _acronymize_database(src_database_file, dst_database_file, acron_dict[src_database_file.stem])

    print("acroynmizing examples ...")
    example_file_rel_paths = ["train_spider.json",
                              "train_others.json",
                              "dev.json",]

    for example_file_rel_path in tqdm(example_file_rel_paths):
        src_file_path = src_dataset_root / example_file_rel_path
        dst_file_path = dst_dataset_root / example_file_rel_path
        _acronymize_examples(src_file_path, dst_file_path, acron_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dataset_root", type=Path, default='/mnt/sdc/jjkim/NL2QGM/data/spider')
    parser.add_argument("--dst_dataset_root", type=Path, default='/mnt/sdc/jjkim/NL2QGM/data/spider-acron')
    args = parser.parse_args()

    src_dataset_root = args.src_dataset_root
    dst_dataset_root = args.dst_dataset_root

    acronymize(src_dataset_root, dst_dataset_root)
