import os
from pathlib import Path
import attr
import json
import time
import pickle
import argparse
import asdl
import psycopg2
import re

from socket import *
from nltk.tokenize import word_tokenize
from ratsql.datasets.utils.db_utils import create_db_conn_str

from run_all import Preprocessor
from ratsql import ast_util
from ratsql.commands.infer import Inferer
from ratsql.datasets.spider_lib.process_sql_postgres import get_sql, get_schema, Schema
from ratsql.utils import registry, evaluation
from ratsql.grammars.spider import SpiderUnparser
from ratsql.grammars.postgres import PostgresUnparser

OPNAME_TO_OP = {
    'Eq': '=',
    'Gt': '>',
    'Lt': '<',
    'Ge': '>=',
    'Le': '<=',
    'Ne': '!='
}

DB_SCHEMA = {}

POSTGRES_CURSOR = None

@attr.s
class InferConfig:
    mode = attr.ib()
    section = attr.ib(default='val')
    beam_size = attr.ib(default=1)
    limit = attr.ib(default=None)
    use_heuristic = attr.ib(default=True)
    output_history = attr.ib(default=False)

@attr.s
class Example:
    question = attr.ib()
    db_id = attr.ib()
    gold = attr.ib(default='')
    dataset_name = attr.ib(default='spider')
    cem = attr.ib(default=[]) # column exact match
    tem = attr.ib(default=[]) # table exact match
    cpm = attr.ib(default=[]) # column partial match
    tpm = attr.ib(default=[]) # table partial match
    cm = attr.ib(default=[])  # cell match
    nm = attr.ib(default=[]) # number match
    dm = attr.ib(default=[]) # date match
    # Exclude
    cem_exclude = attr.ib(default=[]) # column exact match
    tem_exclude = attr.ib(default=[]) # table exact match
    cpm_exclude = attr.ib(default=[]) # column partial match
    tpm_exclude = attr.ib(default=[]) # table partial match
    cm_exclude = attr.ib(default=[])  # cell match
    nm_exclude = attr.ib(default=[]) # number match
    dm_exclude = attr.ib(default=[]) # date match

    @property
    def manual_linking_info(self):
        return {
            "CEM": self.cem,
            "TEM": self.tem,
            "CPM": self.cpm,
            "TPM": self.tpm,
            "CM": self.cm,
            "NM": self.nm,
            "DM": self.dm,
            "CEM_exclude": self.cem_exclude,
            "TEM_exclude": self.tem_exclude,
            "CPM_exclude": self.cpm_exclude,
            "TPM_exclude": self.tpm_exclude,
            "CM_exclude": self.cm_exclude,
            "NM_exclude": self.nm_exclude,
            "DM_exclude": self.dm_exclude,
        }


def modify_file(config, question, db_id, gold=None, save_schema=True, db_type='sqlite', grammar='spider'):
    assert len(config['data']['val']['paths']) == 1
    file_path = config['data']['val']['paths'][0]
    table_path = config['data']['val']['tables_paths'][0]
    db_dir = config['data']['val']['db_path']

    # Check valid db_id
    with open(table_path, 'r') as f:
        table_data = json.load(f)
    dbs = {item['db_id']: item for item in table_data}
    assert db_id in dbs, f"{db_id} is not a valid db_id"
    # # Read in data file
    # with open(file_path, 'r') as f:
    #     data = json.load(f)
    # Modify data
    data = {}
    data['question'] = question
    data['question_toks'] = word_tokenize(question)
    data['db_id'] = db_id
    if gold:
        data['query'] = gold
        data['query_toks'] = word_tokenize(gold)
    # Parse SQL into spider json format
    db_conn_str = create_db_conn_str(db_dir, db_id, db_type=db_type)
    if db_type == 'sqlite':
        assert os.path.isfile(db_conn_str), f'{db_conn_str} does not exist!'
    schema = Schema(get_schema(db_conn_str, db_type=db_type), dbs[db_id])

    if save_schema:
        schema_path = os.path.join(os.path.dirname(__file__), 'saved_schema')
        if not os.path.isdir(schema_path):
            os.mkdir(schema_path)
        
        with open(os.path.join(schema_path, db_id), 'wb') as f:
            pickle.dump(schema, f, pickle.HIGHEST_PROTOCOL)

    if gold:
        g_sql = get_sql(schema, gold)
        data['sql'] = g_sql
    else:
        data['sql'] = ''

    # Overwrite data file
    with open(file_path, 'w') as f:
        f.write(json.dumps([data], indent=4))
    return None


def get_model_from_path(path, step=None, dataset_name='w3schools_test', db_type='sqlite'):
    """
    path: logdir of model
    """
    assert os.path.isdir(path)
    # Get files
    files = [file for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))]

    # Get config
    config_files = [file for file in files if file.startswith('config') and file.endswith('.json')]
    assert len(config_files) == 1, f'Error: More than one config file'
    config = json.load(open(os.path.join(path, config_files[0])))

    # JHCHO, Sep 29, 2021
    # Modify for changing db path used in encoder preprocessing
    dataset_root = Path(f"data/{dataset_name}_test")
    if db_type == 'sqlite':
        db_path = dataset_root / "database"
    elif db_type == 'postgres':
        db_path = 'user=postgres password=postgres host=localhost port=5435'
    else:
        raise KeyError(f'wrong db type: {db_type}')
    config['model']['encoder_preproc']['db_path'] = str(db_path)
    config['model']['decoder_preproc']['db_path'] = str(db_path)
    config['model']['decoder_preproc']['db_type'] = db_type    
    config['model']['encoder_preproc']['save_path'] = str(dataset_root / 'saved_model')
    config['model']['decoder_preproc']['save_path'] = str(dataset_root / 'saved_model')
    config['data']['val']['db_path'] = str(db_path)
    config['data']['val']['db_type'] = db_type
    config['data']['val']['name'] = 'spider'
    config['data']['val']['paths'][0] = str(dataset_root / "dev.json")
    config['data']['val']['tables_paths'][0] = str(dataset_root / "tables.json")


    # Declare model
    inferer = Inferer(config)
    model, _ = inferer.load_model(path, step)

    model.config = config

    return model


def test_example(model, config, inferer, preprocessor, example, dir_path, dataset_name='w3schools_test', db_type='sqlite'):
    start_time = time.time()

    # Modify test example
    print("modify test example")
    modify_file(config, example.question, example.db_id, example.gold, db_type=db_type)
    print(f"Done {time.time() - start_time}")
    
    # Preprocess
    preprocessor.preprocess(example.manual_linking_info)
    
    # Infer
    print("infer ready")
    if inferer is None:
        inferer = Inferer(config)
    print(f"Doing {time.time() - start_time}")
    infer_args = InferConfig(mode='infer')
    # debug_args = InferConfig(mode='debug')
    print(f"Doing {time.time() - start_time}")

    dataset_schema = registry.construct('dataset', config['data']['val'])[0].schema
    print(f"Doing {time.time() - start_time}")
    
    # Change output path
    log_path = f"{dir_path}/ie_dirs"
    if not os.path.isdir(log_path):
        os.makedirs(log_path)
    infer_output_path = os.path.join(log_path, "test-infer.jsonl")
    # debug_output_path = os.path.join(log_path, "test-debug.jsonl")
    print(f"Done {time.time() - start_time}")
    
    # Run inferer mode=infer
    print("infer")
    inferer.infer(model, infer_output_path, infer_args)

    inferred_line = list(open(infer_output_path))[0]
    print(f"Done {time.time() - start_time}")
    print("ready to output")
    inferred_result = json.loads(inferred_line)
    assert 'beams' in inferred_result
    inferred_sql = inferred_result['beams'][0]['inferred_code']
    inferred_schema = inferred_result['schema']
    inferred_sql_dict = inferred_result['beams'][0]['model_output']

    # # Debug
    # inferer.infer(model, debug_output_path, debug_args)

    # # Eval
    # eval_output_path = os.path.join(log_path, 'test-eval.json')

    # inferred = open(infer_output_path)
    # data = registry.construct('dataset', config['data']['val'])
    # metrics = data.Metrics(data, config['data']['val']['tables_paths'][0])

    # inferred_lines = list(inferred)
    # if len(inferred_lines) < len(data):
    #     data = evaluation.filter_examples(inferred_lines, data)

    # for line in inferred_lines:
    #     infer_results = json.loads(line)
    #     assert 'beams' in infer_results
    #     inferred_code = infer_results['beams'][0]['inferred_code']
    #     assert 'index' in infer_results
    #     result = metrics.add(data[infer_results['index']], inferred_code)

    # with open(eval_output_path, 'w') as f:
    #     f.write(json.dumps(metrics.finalize(), indent=4))
    
    # print(f'Wrote eval results to {eval_output_path}')   

    inferred_sql_dict = infer_from_clause(inferred_sql_dict, dataset_schema, grammar=config['model']['decoder_preproc']['grammar']['name'])
    print(f"Done {time.time() - start_time}")

    return inferred_sql, inferred_schema, inferred_sql_dict


def infer_value_from_question(question, table, column):
    global POSTGRES_CURSOR
    search_query = f"SELECT {column} FROM {table}"
    POSTGRES_CURSOR.execute(search_query)
    results = POSTGRES_CURSOR.fetchall()
    if not results:
        return "value_not_found"
    for result in results:
        if result[0] in question:
            return result[0]

    return results[0][0]


def build_where_tree(cond, tables, columns, col_to_tab, infer_value=False, question=None):
    if cond['_type'] in ['And', 'Or']:
        left_cond = cond['left']
        left_cond_list = build_where_tree(left_cond, tables, columns, col_to_tab, infer_value=infer_value)

        right_cond = cond['right']
        right_cond_list = build_where_tree(right_cond, tables, columns, col_to_tab, infer_value=infer_value)

        return [left_cond_list, cond['_type'].upper(), right_cond_list]

    elif cond['_type'] in OPNAME_TO_OP.keys():
        col_id = cond['val_unit']['col_unit1']['col_id']
        col_name = columns[col_id]
        tab_name = tables[col_to_tab[col_id]]
        value = cond['val1']['_type']
        if value == 'Terminal' and infer_value:
            assert question is not None
            value = infer_value_from_question(question, tab_name, col_name)
        formatted_cond = [tab_name + '.' + col_name, OPNAME_TO_OP[cond['_type']], "'" + value + "'"]

        return formatted_cond


def build_where_list_only_and(cond, tables, columns, col_to_tab, infer_value=False, question=None):
    if cond['_type'] in ['And', 'Or']:
        left_cond = cond['left']
        left_cond_list = build_where_list_only_and(left_cond, tables, columns, col_to_tab, infer_value=infer_value)

        right_cond = cond['right']
        right_cond_list = build_where_list_only_and(right_cond, tables, columns, col_to_tab, infer_value=infer_value)

        return left_cond_list + right_cond_list

    elif cond['_type'] in OPNAME_TO_OP.keys():
        col_id = cond['val_unit']['col_unit1']['col_id']
        col_name = columns[col_id]
        tab_name = tables[col_to_tab[col_id]]
        value = cond['val1']['_type']
        if value == 'Terminal' and infer_value:
            assert question is not None
            value = infer_value_from_question(question, tab_name, col_name)
        formatted_cond = [tab_name + '.' + col_name, OPNAME_TO_OP[cond['_type']], "'" + value + "'"]

        return [formatted_cond]


def build_join_cond_list_only_and(cond, tables, columns, col_to_tab):
    if cond['_type'] in ['And', 'Or']:
        left_cond = cond['left']
        left_cond_list = build_join_cond_list_only_and(left_cond, tables, columns, col_to_tab)

        right_cond = cond['right']
        right_cond_list = build_join_cond_list_only_and(right_cond, tables, columns, col_to_tab)

        return left_cond_list + right_cond_list

    elif cond['_type'] in OPNAME_TO_OP.keys():
        col_id1 = cond['val_unit']['col_unit1']['col_id']
        col_name1 = columns[col_id1]
        tab_name1 = tables[col_to_tab[col_id1]]

        col_id2 = cond['val1']['c']['col_id']
        col_name2 = columns[col_id2]
        tab_name2 = tables[col_to_tab[col_id2]]

        formatted_cond = [tab_name1 + '.' + col_name1 + OPNAME_TO_OP[cond['_type']] + tab_name2 + '.' + col_name2]

        return [formatted_cond]


def infer_from_clause(sql_dict, schema, grammar='postgres'):
    if grammar in {'postgres', 'postgresql'}:
        asdl_file = "Postgres.asdl"
    elif grammar in {'spider'}:
        asdl_file = "Spider_f2.asdl"
    else:
        raise KeyError(f'wrong grammar name: {grammar}')
    custom_primitive_type_checkers = {}
    custom_primitive_type_checkers['table'] = lambda x: isinstance(x, int)
    custom_primitive_type_checkers['column'] = lambda x: isinstance(x, int)

    ast_wrapper = ast_util.ASTWrapper(
            asdl.parse(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    'ratsql',
                    'grammars',
                    asdl_file)),
            custom_primitive_type_checkers=custom_primitive_type_checkers)

    if grammar in {'postgres', 'postgresql'}:
        unparser = PostgresUnparser(ast_wrapper, schema, 2, do_refine_from=True)
    elif grammar in {'spider'}:
        unparser = SpiderUnparser(ast_wrapper, schema, 2, do_refine_from=True)
    else:
        raise KeyError(f'wrong grammar name: {grammar}')
    unparser.refine_from(sql_dict)

    return sql_dict


# Jhcho - 21.11.05: Fix schema issue
def build_schema_dict(dataset_name):
    global DB_SCHEMA
    tables_file = json.load(open(os.path.join('data', dataset_name, 'tables.json')))
    for ent in tables_file:
        db_id = ent['db_id']
        table_names = ent['table_names_original']
        column_names = []
        col_to_tab = {}
        for col_id, (tab_id, col_name) in enumerate(ent['column_names_original']):
            column_names.append(col_name)
            col_to_tab[col_id] = tab_id
        DB_SCHEMA[db_id] = {
                'tables': table_names,
                'columns': column_names,
                'col_to_tab': col_to_tab
        }


def format_result(sql, sql_dict, schema, db_id, infer_value=False, question=None):
    # tables = schema[db_id]['table_names_original']
    # columns = schema[db_id]['column_names_original']

    # for column in columns:
    #     tab_id, col_name = column
    #     if tab_id != -1:
    #         tab_name = tables[tab_id]
    #         schema_dict[tab_name].append(col_name)

    # Jhcho - 21.11.05: fix schema issue
    # tables = [''.join([tab[0][1:]] + tab[1:]) for tab in schema['tables']]
    # columns = [''.join([col[0][1:]] + col[1:-1]) for col in schema['columns']]
    # col_to_tab = schema['column_to_table']

    global DB_SCHEMA
    tables = DB_SCHEMA[db_id]['tables']
    columns = DB_SCHEMA[db_id]['columns']
    col_to_tab = DB_SCHEMA[db_id]['col_to_tab']

    schema_dict = {k: [] for k in tables}
    for col_id, col_name in enumerate(columns):
        tab_id = col_to_tab[col_id]
        if not tab_id:
            continue
        tab_name = tables[tab_id]
        schema_dict[tab_name].append(col_name)

    select_list = []
    for agg in sql_dict['select']['aggs']:
        agg_type = agg['agg_id']['_type']
        if agg_type == 'NoneAggOp':
            agg_name = 'NONE'
        else:
            agg_name = agg_type.upper()
        col_id = agg['val_unit']['col_unit1']['col_id']
        tab_name = tables[col_to_tab[col_id]]
        col_name = columns[col_id]
        select_list.append([agg_name, tab_name + '.' + col_name])

    from_list = []
    for tab in sql_dict['from']['table_units']:
        tab_id = tab['table_id']
        tab_name = tables[tab_id]
        from_list.append(tab_name)
    
    join_cond_list = []
    if 'conds' in sql_dict['from']:
        join_cond = sql_dict['from']['conds']
        # join_cond_list = build_where_tree(join_cond, tables, columns, col_to_tab)
        join_cond_list = build_join_cond_list_only_and(join_cond, tables, columns, col_to_tab) # Accept only and conditions

    where_list = []
    if 'where' in sql_dict['sql_where']:
        where_cond = sql_dict['sql_where']['where']
        # where_list = build_where_tree(where_cond, tables, columns, col_to_tab)
        where_list = build_where_list_only_and(where_cond, tables, columns, col_to_tab, infer_value=infer_value, question=question) # Accept only and conditions

    group_list=[]
    if 'group_by' in sql_dict['sql_groupby']:
        for group in sql_dict['sql_groupby']['group_by']:
            col_id = group['col_id']
            tab_name = tables[col_to_tab[col_id]]
            col_name = columns[col_id]
            group_list.append(tab_name + '.' + col_name)

    # if len(sql_dict['select'][1]) == 1 and sql_dict['select'][1][0][1] == '*':
    #     tab_name = sql_dict['select'][1][0][2]
    #     all_columns = [('none', x, tab_name, False) for x in schema_dict[tab_name]]
    #     sql_dict['select'][1] = all_columns

    formatted_sql_dict = {
        'select': select_list,
        'from': from_list,
        'where': where_list,
        'groupby': group_list,
        'join_conditions': join_cond_list
    }

    result = json.dumps(formatted_sql_dict) + '\n' + db_id + '\n' + json.dumps(schema_dict)

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='/mnt/sdc/jjkim/NL2QGM/logdir/samsung-addop-nopre/bs=1,lr=7.4e-04,bert_lr=3.0e-06,end_lr=0e0,seed=0,join_cond=false')

    args = parser.parse_args()
    model_dir = args.model_dir
    dataset_name = 'samsung'
    db_type = 'postgres'

    pg_config = 'host=localhost port=5435 user=postgres password=postgres dbname=' + dataset_name
    pg_connection = psycopg2.connect(pg_config)
    POSTGRES_CURSOR = pg_connection.cursor()

    default_query = "SELECT * FROM customers" if dataset_name == 'w3schools_test' else "SELECT * FROM quality.m_fab_dcop_met"

    build_schema_dict(dataset_name)
    
    start_time = time.time()
    model = get_model_from_path(model_dir, dataset_name=dataset_name, db_type=db_type)

    def modify_config(config):
        # Modify data config
        if 'train' in config['data']:
            del config['data']['train']
        return config
    config = modify_config(model.config)
    inferer = Inferer(config)
    preprocessor = Preprocessor(config)
    print("Model Load Time: %.2f"%(time.time() - start_time))

    def infer_one(question, gold):
        example = Example(
            question=question,
            gold=default_query,
            db_id='samsung',
            dataset_name='samsung'
        )
        print("Example made")
        sql, schema, sql_dict = test_example(model, config, inferer, preprocessor, example,
                                            dir_path=model_dir,
                                            dataset_name=dataset_name, db_type=db_type)
        print(f"Question: {question}")
        print(f"Pred: {sql}")
        print(f"Gold: {gold}")

    infer_one("PPIT 공법ID에 대한 모든 Root lot ID 개수 조회", "select count (distinct root_lot_id) from QUALITY.m_fab_tracking where PROCESS_ID = 'PPIT'")
    infer_one("KHAA84901B-GEL 제품에 속하는 LOT에 대해 검출된 DEFECT 전체 개수의 최대값을 계산", "select lot_id, MAX(total_dft_cnt) from quality.m_defect_chip where part_id = KHAA84901B-GEL group by LOT_ID")
    infer_one("KCAK 공법에 속한 LOT, WAFER에 대해 EDS CHIP BIN 테스트 대상 CHIP 개수를 조회", "select lot_id, wafer_id, array_length(BIN_NO,1) from quality.m_eds_chip_bin where process_id = 'KCAK'")
    infer_one("PART ID가 K9GDGD8U0D-CXT이고, ROOT LOT ID가 CKEBB7인 LOT, WAFER에 대하여 FAB VALUE를 조회", "select lot_id, wafer_id, unnest(fab_value) fab_value from quality.m_fab_dcop_met where part_id = 'K9GDGD8U0D-CXT' and root_lot_id = 'CKEBB7'")
    infer_one("KFBG 라인 GKG706 root lot에 해당하는 설비 id, DCOP 계측 아이템 id를 조회하시오", "select eqp_id, unnest(item_id) as itemID from quality.m_fab_dcop_met where line_id = 'KFBG' and root_lot_id = 'GKG706'")
    infer_one("KFBH 라인에서 수행하는 KGKV 제품의 총 DEFECT TEST 항목 개수를 조회", "select count (distinct (step_seq||','||item_id)) from master.m_defect_item where line_id = 'KFBH' and process_id = 'KGKV'")



