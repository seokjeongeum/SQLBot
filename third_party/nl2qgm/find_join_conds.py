import os
import pickle
import json
import argparse

from ratsql.datasets.spider_lib.process_sql_postgres import get_sql
from ratsql.grammars.spider import SpiderLanguage
from table_search import format_result, build_schema_dict

def find_join_condition(processed_tables, remaining_tables, join_conds, schema):
    if len(remaining_tables) == 0:
        return join_conds

    for fk, pk in schema['foreign_keys']:
        if schema['column_names'][fk][0] in processed_tables and schema['column_names'][pk][0] in remaining_tables:
            remaining_tables.remove(schema['column_names'][pk][0])
            processed_tables.append(schema['column_names'][pk][0])
            fcol_name = schema["column_names_original"][fk][1]
            ftab_name = schema["table_names_original"][schema["column_names_original"][fk][0]]
            pcol_name = schema["column_names_original"][pk][1]
            ptab_name = schema["table_names_original"][schema["column_names_original"][pk][0]]
            join_conds.append(f"{ftab_name}.{fcol_name}={ptab_name}.{pcol_name}")
        if schema['column_names'][pk][0] in processed_tables and schema['column_names'][fk][0] in remaining_tables:
            remaining_tables.remove(schema['column_names'][fk][0])
            processed_tables.append(schema['column_names'][fk][0])
            fcol_name = schema["column_names_original"][fk][1]
            ftab_name = schema["table_names_original"][schema["column_names_original"][fk][0]]
            pcol_name = schema["column_names_original"][pk][1]
            ptab_name = schema["table_names_original"][schema["column_names_original"][pk][0]]
            join_conds.append(f"{ftab_name}.{fcol_name}={ptab_name}.{pcol_name}")
    return find_join_condition(processed_tables, remaining_tables, join_conds, schema)



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_id', type=str, default='w3schools_test')
    parser.add_argument('--tables', type=str, required=True)

    args = parser.parse_args()
    tables = args.tables.split(",")
    tables_file = json.load(open(os.path.join('data', "samsung_test", 'tables.json')))
    schema = None
    for ent in tables_file:
        if ent["db_id"] == args.db_id:
            schema = ent
            break
    table_ids = [i for i in range(len(schema["table_names_original"])) if schema["table_names_original"][i] in tables]
    result = find_join_condition([table_ids[0]], table_ids[1:], [], schema)
    if result:
        print(" AND ".join(result))
    else:
        print("None")


       
