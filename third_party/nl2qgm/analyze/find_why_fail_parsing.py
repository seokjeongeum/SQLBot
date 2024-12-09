import os
import json

from dblab_preprocess.process_sql import get_sql, tokenize
from dblab_preprocess.parse_queries import Schema, get_schemas_from_json

target_file_path = "/root/NL2QGM/data/wikitablequestions/test.json"
tables_path = "/root/NL2QGM/data/wikitablequestions/tables.json"

if __name__ == "__main__":
    data = json.load(open(target_file_path))
    schemas, db_names, tables = get_schemas_from_json(tables_path)
    
    e_list = {}
    for idx, datum in enumerate(data):
        db_name = datum['table_names'][0].replace(" ", "_")
        schema = schemas[db_name]
        table = tables[db_name]
        schema = Schema(schema, table)
        query = datum['query']
        try:
            sql = get_sql(schema, query, table)
        except Exception as e:
            print(f"\nIdx:{idx}")
            print(f"Exception: {e}")
            print(f"Query: {query}")
            if e not in e_list:
                e_list[e] = 1
            else:
                e_list[e] += 1
            # sql = get_sql(schema, query, table)
    stop = 1
