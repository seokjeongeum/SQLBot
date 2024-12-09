import os
import json
import tqdm
import sqlite3

"""
This is based on the newly created tables.json (i.e. tables_extended.json)
"""
tables_path = "/root/NL2QGM/data/spider/tables_extended.json"
db_dir = "/root/NL2QGM/data/spider/database/"


class ConnectDB:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None

    def __enter__(self):
        # Create dirctory if not exist
        db_dir = '/'.join(self.db_path.split('/')[:-1])
        if not os.path.exists(db_dir):
            os.makedirs(db_dir)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.text_factory = lambda b: b.decode(errors = 'ignore')
        return self.conn.cursor()

    def __exit__(self, exception_type, exception_val, trace):
        self.conn.commit()
        self.conn.close()


def pprint(tqdm_instance, message):
    if tqdm_instance:
        tqdm_instance.set_description(message)
        tqdm_instance.refresh()
    else:
        print(message)


def load_db_schema_queries(db_name):
    db_path = os.path.join(db_dir, db_name, f"{db_name}.sqlite")
    assert os.path.exists(db_path), f"{db_path} doest not exist!"

    with ConnectDB(db_path) as cursor:
        schema = cursor.execute("SELECT * FROM sqlite_master;").fetchall()

    create_queries = []
    for item in schema:
        if item[0] == 'table':
            # Change bad table name
            if item[1] == "sqlite_sequence":
                continue
            create_queries.append(item[4])
    return create_queries


def load_db_records_queries(db_name):
    db_path = os.path.join(db_dir, db_name, f"{db_name}.sqlite")
    assert os.path.exists(db_path), f"{db_path} does not exist!"
    insert_queries = []

    with ConnectDB(db_path) as cursor:
        tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
        for table_name in map(lambda k: k[0], tables):
            column_names = cursor.execute(f"SELECT * from {table_name}").description
            column_names = list(map(lambda k:k [0], column_names))
            
            records = cursor.execute(f"SELECT * from {table_name}").fetchall()
            for record in records:
                # Filter Null values
                non_null_columns = []
                non_null_values = []
                for idx in range(len(record)):
                    if record[idx] != None:
                        non_null_columns.append(column_names[idx])
                        non_null_values.append(record[idx])
                non_null_values = tuple(non_null_values)
                non_null_columns = tuple(non_null_columns)

                # Insert query
                if table_name == "sqlite_sequence":
                    continue
                insert_queries.append(f"insert into {table_name} {non_null_columns} values {non_null_values};")
    
    return insert_queries


def create_db(db_name, src_dbs, tqdm_instance=None):
    pprint(tqdm_instance, f"Creating {db_name}...")

    db_path = os.path.join(db_dir, db_name, f"{db_name}.sqlite")
    if os.path.exists(db_path):
        pprint(tqdm_instance, f"{db_path} already exist!")
        return None

    create_table_queries = []
    insert_record_queries = []
    for src_db_id in src_dbs:
        # Load schemas
        create_table_queries += load_db_schema_queries(src_db_id)

        # Load records
        insert_record_queries += load_db_records_queries(src_db_id)

    with ConnectDB(db_path) as cursor:
        # Create schema
        pprint(tqdm_instance, f"Creating schema for {db_name}...")
        for create_table_query in create_table_queries:
            cursor.execute(create_table_query)
        
        # Append records
        pprint(tqdm_instance, f"Inserting records for {db_name}...")
        for insert_query in insert_record_queries:
            cursor.execute(insert_query)
    pprint(tqdm_instance, f"Done creating: {db_name}!")


def validate_db(db_name, db_info, tqdm_instance=None):
    pprint(tqdm_instance, f"Validating {db_name}...")
    db_path = os.path.join(db_dir, db_name, f"{db_name}.sqlite")
    assert os.path.exists(db_path), f"{db_path} does not exist!"

    # Load all tables
    with ConnectDB(db_path) as cursor:
        schema = cursor.execute("SELECT * FROM sqlite_master;").fetchall()
    table_items = filter(lambda k: k[0] == 'table', schema)
    table_names = list(map(lambda k: k[1], table_items))

    # Compare table names
    assert len(db_info['table_names_original']) == len(table_names), \
            "{len(db_info['table_names_original'])} {len(table_names)}"
    assert set(table_names) == set(db_info['table_names_original']), \
            "Check table.. something is wrong"
    
    # Load all columns
    column_names = []
    with ConnectDB(db_path) as cursor:
        tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
        for table_name in map(lambda k: k[0], tables):
            local_column_names = cursor.execute(f"SELECT * from {table_name}").description
            column_names += list(map(lambda k:k [0], local_column_names))
    
    # Compare column names
    assert len(db_info['column_names_original'])-1 == len(column_names), \
        f"{len(db_info['column_names_original'])-1} {len(column_names)}"
    assert set(column_names) == set([item[1] for item in db_info['column_names_original'][1:]]), \
        "Check column.. Something is wrong"


if __name__ == "__main__":
    # Load tables file
    tables = json.load(open(tables_path))
    tables = {item['db_id']: item for item in tables}

    filtered_tables = dict(filter(lambda k: '__' in k[0], tables.items()))

    tqdm_instance = tqdm.tqdm(filtered_tables.items())
    for db_id, item in tqdm_instance:
        create_db(db_id, item['db_id_list'], tqdm_instance=tqdm_instance)

    # Assert created DBs are good
    tqdm_instance = tqdm.tqdm(filtered_tables.items())
    for db_id, item in tqdm_instance:
        validate_db(db_id, item, tqdm_instance=tqdm_instance)

    print("Done!")
