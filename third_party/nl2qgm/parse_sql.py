import os
import pickle
import argparse

from ratsql.datasets.spider_lib.process_sql import get_sql as get_sql_spider
from ratsql.datasets.spider_lib.process_sql_postgres import get_sql as get_sql_postgres
from ratsql.datasets.spider_lib.process_sql import get_sql
from ratsql.grammars.spider import SpiderLanguage
from ratsql.grammars.postgres import PostgresLanguage
from table_search import format_result, build_schema_dict


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_id', type=str, default='w3schools_test')
    parser.add_argument('--query', type=str, required=True)
    parser.add_argument('--grammar', type=str, default='spider')

    args = parser.parse_args()

    schema_path = os.path.join(os.path.dirname(__file__), 'saved_schema')
    schema = pickle.load(open(os.path.join(schema_path, args.db_id), 'rb'))

    if args.grammar == 'spider':
        sql = get_sql_spider(schema, args.query)
    elif args.grammar == 'postgres':
        sql = get_sql_postgres(schema, args.query)

    sl = SpiderLanguage(
        output_from=True,
        use_table_pointer=True,
        include_literals=True,
        include_columns=True,
        end_with_from=True,
        infer_from_conditions=True,
        factorize_sketch=2
    )
    pl = PostgresLanguage(
        output_from=True,
        use_table_pointer=True,
        include_literals=True,
        include_columns=True,
        end_with_from=True,
        infer_from_conditions=True,
        factorize_sketch=2
    )

    if args.grammar == 'spider':
        parsed_sql_dict = sl.parse(sql, 'val')
    elif args.grammar == 'postgres':
        parsed_sql_dict = pl.parse(sql, 'val')

    build_schema_dict("samsung_test")
    result = format_result(args.query, parsed_sql_dict, {}, args.db_id, infer_value=False, grammar=args.grammar)

    print(result)

       
