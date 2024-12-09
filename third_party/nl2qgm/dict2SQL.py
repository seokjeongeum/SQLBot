import os
import asdl
import argparse
import json
import pickle

from ratsql import ast_util
from ratsql.grammars.spider import SpiderUnparser

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--use_table_pointer', action='store_true')
    parser.add_argument('--include_columns', action='store_true')
    parser.add_argument('--factorize_sketch', type=int, default=2)
    parser.add_argument('--sql_dict', type=str, required=True)
    parser.add_argument('--db_id', type=str, required=True)

    args = parser.parse_args()

    custom_primitive_type_checkers = {}
    if args.use_table_pointer:
        custom_primitive_type_checkers['table'] = lambda x: isinstance(x, int)
    if args.include_columns:
        custom_primitive_type_checkers['column'] = lambda x: isinstance(x, int)

    if args.factorize_sketch == 0:
        asdl_file = "Spider.asdl"
    elif args.factorize_sketch == 1:
        asdl_file = "Spider_f1.asdl"
    elif args.factorize_sketch == 2:
        asdl_file = "Spider_f2.asdl"
    else:
        raise NotImplementedError
    
    ast_wrapper = ast_util.ASTWrapper(
        asdl.parse(
            os.path.join(
                os.path.dirname(__file__), 'ratsql', 'grammars',
                asdl_file)),
        custom_primitive_type_checkers=custom_primitive_type_checkers)

    schema_path = os.path.join(os.path.dirname(__file__), 'saved_schema', args.db_id)
    with open(schema_path, 'rb') as f:
        schema = pickle.load(f)

    unparser = SpiderUnparser(ast_wrapper, schema, args.factorize_sketch, do_refine_from=True)

    test = '{"_type": "sql", "select": {"_type": "select", "is_distinct": false, "aggs": [{"_type": "agg", "agg_id": {"_type": "NoneAggOp"}, "val_unit": {"_type": "Column", "col_unit1": {"_type": "col_unit", "agg_id": {"_type": "NoneAggOp"}, "col_id": 41, "is_distinct": false}}}]}, "sql_where": {"_type": "sql_where"}, "sql_groupby": {"_type": "sql_groupby", "group_by": [{"_type": "col_unit", "agg_id": {"_type": "NoneAggOp"}, "col_id": 2, "is_distinct": false}]}, "sql_orderby": {"_type": "sql_orderby", "order_by": {"_type": "order_by", "order": {"_type": "Desc"}, "val_units": [{"_type": "Column", "col_unit1": {"_type": "col_unit", "agg_id": {"_type": "Count"}, "col_id": 0, "is_distinct": false}}]}, "limit": true}, "sql_ieu": {"_type": "sql_ieu"}, "from": {"_type": "from", "table_units": [{"_type": "Table", "table_id": 0}, {"_type": "Table", "table_id": 5}, {"_type": "Table", "table_id": 7}], "conds": {"_type": "And", "left": {"_type": "Eq", "val_unit": {"_type": "Column", "col_unit1": {"_type": "col_unit", "agg_id": {"_type": "NoneAggOp"}, "col_id": 1, "is_distinct": false}}, "val1": {"_type": "ColUnit", "c": {"_type": "col_unit", "agg_id": {"_type": "NoneAggOp"}, "col_id": 29, "is_distinct": false}}}, "right": {"_type": "Eq", "val_unit": {"_type": "Column", "col_unit1": {"_type": "col_unit", "agg_id": {"_type": "NoneAggOp"}, "col_id": 28, "is_distinct": false}}, "val1": {"_type": "ColUnit", "c": {"_type": "col_unit", "agg_id": {"_type": "NoneAggOp"}, "col_id": 35, "is_distinct": false}}}}}}'

    tree = json.loads(test)#args.sql_dict)
    sql = unparser.unparse_sql(tree)

    print(sql)