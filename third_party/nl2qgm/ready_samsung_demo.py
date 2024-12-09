# Run this file after running postgreSQL server by docker image
# What this script do?
# make tables.json file and make saved_schema picke file (Even though I don't know why we have to maintain two files..)
import pickle

import os
from ratsql.datasets.utils.db_to_tables import db_to_tables
from pathlib import Path
import json
from ratsql.datasets.spider_lib.process_sql_postgres import get_sql, get_schema, Schema
from ratsql.datasets.utils.db_utils import create_db_conn_str


def main():
    if not os.path.exists("data/samsung_test"):
        os.mkdir("data/samsung_test")
    if not os.path.exists("saved_schema"):
        os.mkdir("saved_schema")
    db_to_tables(Path("data/samsung_test/tables.json"))
    with open("data/samsung_test/tables.json") as f:
        table_data = json.load(f)
    for table_datum in table_data:
        db_conn_str = create_db_conn_str('user=postgres password=postgres host=localhost port=5435', table_datum["db_id"], db_type="postgres")

        schema = Schema(get_schema(db_conn_str, db_type="postgres"), table_datum)
        with open(f"saved_schema/{table_datum['db_id']}", "wb") as f:
            pickle.dump(schema, f)


if __name__ == "__main__":
    main()
