import logging
import traceback

import os
import tqdm
import sqlite3
from pathlib import Path
import attr
import json
import time
import pickle
import argparse
import asdl
import psycopg2
import re
import torch
import _jsonnet

from socket import *
from nltk.tokenize import word_tokenize
from ratsql.datasets.utils.db_utils import create_db_conn_str

from run_all import Preprocessor
from ratsql import ast_util
from ratsql.commands.infer import Inferer
from ratsql.commands.analysis import Attribution
from ratsql.datasets.spider_lib.process_sql_postgres import get_sql, get_schema, Schema
from ratsql.utils import registry, evaluation
from ratsql.grammars.spider import SpiderLanguage, SpiderUnparser
# from ratsql.grammars.postgres import PostgresLanguage, PostgresUnparser
from ratsql.models.spider.spider_enc import SpiderEncoderBertPreproc, Bertokens
from ratsql.models.spider import spider_beam_search
from ratsql.datasets.spider import load_tables, SpiderItem

from ratsql.models.cosql.cosql_enc import CosqlEncoderBert
from ratsql.models.nl2intent.decoder import NL2IntentDecoder

OPNAME_TO_OP = {
    "Eq": "=",
    "Gt": ">",
    "Lt": "<",
    "Ge": ">=",
    "Le": "<=",
    "Ne": "!=",
    "Between": "BETWEEN",
    "In": "IN",
    "Like": "LIKE",
    "Is": "IS",
}

DB_SCHEMA = {}

POSTGRES_CURSOR = None

os.environ['CUDA_VISIBLE_DEVICES']='2'
@attr.s
class InferConfig:
    mode = attr.ib()
    device = attr.ib()
    section = attr.ib(default="val")
    beam_size = attr.ib(default=5)
    limit = attr.ib(default=None)
    use_heuristic = attr.ib(default=True)
    output_history = attr.ib(default=False)


@attr.s
class Example:
    question = attr.ib()
    db_id = attr.ib()
    gold = attr.ib(default="")
    dataset_name = attr.ib(default="spider")
    cem = attr.ib(default=[])  # column exact match
    tem = attr.ib(default=[])  # table exact match
    cpm = attr.ib(default=[])  # column partial match
    tpm = attr.ib(default=[])  # table partial match
    cm = attr.ib(default=[])  # cell match
    nm = attr.ib(default=[])  # number match
    dm = attr.ib(default=[])  # date match
    # Exclude
    cem_exclude = attr.ib(default=[])  # column exact match
    tem_exclude = attr.ib(default=[])  # table exact match
    cpm_exclude = attr.ib(default=[])  # column partial match
    tpm_exclude = attr.ib(default=[])  # table partial match
    cm_exclude = attr.ib(default=[])  # cell match
    nm_exclude = attr.ib(default=[])  # number match
    dm_exclude = attr.ib(default=[])  # date match

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

class One_time_Preprocesser():
    def __init__(self, db_path, table_path, preproc_args):
        self.enc_preproc = SpiderEncoderBertPreproc(**preproc_args)
        self.bert_version = preproc_args['bert_version']
        self.schemas = load_tables([table_path], True)[0]
        self._conn(db_path)
    def _conn(self, db_path):
        # Backup in-memory copies of all the DBs and create the live connections
        for db_id, schema in tqdm.tqdm(self.schemas.items(), desc="DB connections"):
            sqlite_path = Path(db_path) / db_id / f"{db_id}.sqlite"
            source: sqlite3.Connection
            if os.path.isfile(sqlite_path):
                with sqlite3.connect(str(sqlite_path)) as source:
                    dest = sqlite3.connect(':memory:')
                    dest.row_factory = sqlite3.Row
                    source.backup(dest)
                schema.connection = dest
    
    def run(self, text, db_id):
        schema = self.schemas[db_id]
        # Validate 
        question = self.enc_preproc._tokenize(text.split(' '), text)
        preproc_schema = self.enc_preproc._preprocess_schema(schema, bert_version=self.bert_version)
        num_words = len(question) + 2 + \
            sum(len(c) + 1 for c in preproc_schema.column_names) + \
            sum(len(t) + 1 for t in preproc_schema.table_names)
        assert num_words < 512, "input too long"
        question_bert_tokens = Bertokens(question, bert_version=self.bert_version)
        # preprocess
        sc_link = question_bert_tokens.bert_schema_linking(
                preproc_schema.normalized_column_names,
                preproc_schema.normalized_table_names
            )
        cv_link = question_bert_tokens.bert_cv_linking(schema)
        spider_item = SpiderItem(text=text,code=None, schema=self.schemas[db_id], orig=None, orig_schema=self.schemas[db_id].orig)
        preproc_item = {
            'sql': '',
            'raw_question': text,
            'question': question,
            'db_id': schema.db_id,
            'sc_link': sc_link,
            'cv_link': cv_link,
            'columns': preproc_schema.column_names,
            'tables': preproc_schema.table_names,
            'table_bounds': preproc_schema.table_bounds,
            'column_to_table': preproc_schema.column_to_table,
            'table_to_columns': preproc_schema.table_to_columns,
            'foreign_keys': preproc_schema.foreign_keys,
            'foreign_keys_tables': preproc_schema.foreign_keys_tables,
            'primary_keys': preproc_schema.primary_keys,
            'interaction_id': None,
            'turn_id': None
        }
        return spider_item, preproc_item

class One_time_inferer():
    def __init__(self, preproc, inferer):
        self.preproc = preproc
        self.inferer = inferer
    def run(self, text, schema):
        preproc_item = self.preproc.run(text, schema)
        prediction = self.inferer.infer(preproc_item)
        return prediction

class UserIntentModel(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super(UserIntentModel, self).__init__()
        self.encoder = encoder.eval()
        self.decoder = decoder.eval()
    def forward(self, inp):
        # Preprocess input
        enc_features = self.encoder(inp)
        logits = self.decoder.decoder_layers(enc_features)
        pred_ids = logits.argmax(dim=1)
        pred_labels = [self.decoder.output_classes[id] for id in pred_ids]
        return pred_labels[0]
    def change_input_format(self, question):
        question = question[len("<s> "):]
        question = question.replace("<s>", "[CLS]")
        return question

def modify_file(
    config, question, db_id, save_schema=True, db_type="sqlite", grammar="spider"
):
    assert len(config["data"]["val"]["paths"]) == 1
    file_path = config["data"]["val"]["paths"][0]
    table_path = config["data"]["val"]["tables_paths"][0]
    db_dir = config["data"]["val"]["db_path"]

    # Check valid db_id
    with open(table_path, "r") as f:
        table_data = json.load(f)
    dbs = {item["db_id"]: item for item in table_data}
    assert db_id in dbs, f"{db_id} is not a valid db_id"
    # # Read in data file
    # with open(file_path, 'r') as f:
    #     data = json.load(f)
    # Modify data
    data = {}
    data["question"] = question
    data["question_toks"] = word_tokenize(question)
    data["db_id"] = db_id
    # Parse SQL into spider json format
    db_conn_str = create_db_conn_str(db_dir, db_id, db_type=db_type)
    if db_type == "sqlite":
        assert os.path.isfile(db_conn_str), f"{db_conn_str} does not exist!"
    schema = Schema(get_schema(db_conn_str, db_type=db_type), dbs[db_id])

    if save_schema:
        schema_path = os.path.join(os.path.dirname(__file__), "saved_schema")
        if not os.path.isdir(schema_path):
            os.mkdir(schema_path)

        with open(os.path.join(schema_path, db_id), "wb") as f:
            pickle.dump(schema, f, pickle.HIGHEST_PROTOCOL)

    data["sql"] = ""

    # Overwrite data file
    with open(file_path, "w") as f:
        f.write(json.dumps([data], indent=4))
    return None


def get_user_intent_model(path_dic, device, preprocessor):
    # Load infos from path
    with open(path_dic["config"]) as f:
        config = json.load(f)
        encoder_config = config["model"]["encoder"]
        decoder_config = config["model"]["decoder"]
    model_state_dict = torch.load(path_dic["model"], map_location=device)
    with open(path_dic['output']) as f:
        possible_outputs = json.load(f)
    # Create model
    user_intent_encoder = CosqlEncoderBert(device, preprocessor.enc_preproc, 
                                           bert_token_type=encoder_config['bert_token_type'], 
                                           bert_version=encoder_config['bert_version'], 
                                           summarize_header=encoder_config['summarize_header'], 
                                           use_column_type=encoder_config['use_column_type']).eval()
    user_intent_decoder = NL2IntentDecoder(device, possible_outputs, decoder_config['enc_recurrent_size'], 
                                           decoder_config['recurrent_size'], decoder_config['dropout']).eval()
    # Load trained params
    user_intent_encoder.load_state_dict({k[len("encoder."):]:v for k, v in model_state_dict['model'].items() if 'encoder.bert_model' in k})
    user_intent_decoder.load_state_dict({k[len("decoder."):]:v for k,v in model_state_dict['model'].items() if 'decoder.' in k})
    user_intent_model = UserIntentModel(user_intent_encoder, user_intent_decoder)
    return user_intent_model


def get_model_from_path2(
    path,
    device,
    db_path,
    table_path,
):
    # Load config
    if os.path.isfile(os.path.join(path, "config.jsonnet")):
        exp_config = json.loads(_jsonnet.evaluate_file(os.path.join(path, "config.jsonnet")))
        model_config_file = exp_config["model_config"]
        model_config_args = exp_config["model_config_args"]
        config = json.loads(_jsonnet.evaluate_file(model_config_file, tla_codes={'args': json.dumps(model_config_args)}))
    else:
        assert os.path.isfile(os.path.join(path, "config.json")), "config file does not exist"
        config = json.load(open(os.path.join(path, "config.json")))

    # Load config
    if os.path.isfile(os.path.join(path, "config_captum.jsonnet")):
        exp_config_captum = json.loads(_jsonnet.evaluate_file(os.path.join(path, "config_captum.jsonnet")))
        model_config_captum_file = exp_config_captum["model_config"]
        model_config_captum_args = exp_config_captum["model_config_args"]
        config_captum = json.loads(_jsonnet.evaluate_file(model_config_captum_file, tla_codes={'args': json.dumps(model_config_captum_args)}))
    else:
        assert os.path.isfile(os.path.join(path, "config_captum.json")), "config file does not exist"
        config_captum = json.load(open(os.path.join(path, "config_captum.json")))

    # config['model']['encoder_preproc']['save_path'] = path
    # config['model']['decoder_preproc']['save_path'] = path
    # config_captum['model']['encoder_preproc']['save_path'] = path
    # config_captum['model']['decoder_preproc']['save_path'] = path
    # db_path = config['data']['train']['db_path']
    # table_paths = config['data']['train']['tables_paths']
    # Create preprocessor
    preprocessor = One_time_Preprocesser(db_path, table_path, config['model']['encoder_preproc'])
    
    # Create inferer
    inferer = Inferer(config)
    model, step = inferer.load_model(os.path.join(path))
    model.to(device)
    
    inferer_captum = Inferer(config_captum)
    model_captum, step = inferer_captum.load_model(os.path.join(path))
    model_captum.to(device)
    # Test inference
    # preproc_item = preprocessor.run("Show all singers", schema)
    # beams = spider_beam_search.beam_search_with_heuristics(model, None, preprocessor, beam_size=2, max_steps=150)
    
    return model, preprocessor, model_captum, config_captum



def get_model_from_path(
    path,
    device,
    is_kor=False,
    step=None,
    dataset_name="w3schools_test",
    db_type="sqlite",
):
    """
    path: logdir of model
    """
    assert os.path.isdir(path)
    # Get files
    files = [
        file for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))
    ]

    # Get config
    config_files = [
        file for file in files if file.startswith("config") and file.endswith(".json")
    ]
    assert len(config_files) == 1, f"Error: More than one config file"
    config = json.load(open(os.path.join(path, config_files[0])))

    # JHCHO, Sep 29, 2021
    # Modify for changing db path used in encoder preprocessing
    dataset_root = Path(os.path.join("data", dataset_name))

    if db_type == "sqlite":
        db_path = dataset_root / "database"
    elif db_type == "postgres":
        db_path = "user=postgres password=postgres host=localhost port=5435"
    else:
        raise KeyError(f"wrong db type: {db_type}")
    config["model"]["encoder_preproc"]["db_path"] = str(db_path)
    config["model"]["decoder_preproc"]["db_path"] = str(db_path)
    config["model"]["decoder_preproc"]["db_type"] = db_type
    config["model"]["encoder_preproc"]["save_path"] = str(
        dataset_root / ("saved_model_kor" if is_kor else "saved_model")
    )
    config["model"]["decoder_preproc"]["save_path"] = str(
        dataset_root / ("saved_model_kor" if is_kor else "saved_model")
    )
    config["data"]["val"]["db_path"] = str(db_path)
    config["data"]["val"]["db_type"] = db_type
    config["data"]["val"]["name"] = "spider"
    config["data"]["val"]["paths"][0] = str(dataset_root / "dev.json")
    config["data"]["val"]["tables_paths"][0] = str(dataset_root / "tables.json")

    config["model"]["decoder"]["value_vocab_path"] = str(
        dataset_root / "value_vocab.json"
    )

    # Declare model
    inferer = Inferer(config, model_dir=path)
    model, _ = inferer.load_model(path, step)

    model.config = config

    def modify_config(config):
        # Modify data config
        if "train" in config["data"]:
            del config["data"]["train"]
        return config

    device = torch.device(device)
    model.to(device)
    config = modify_config(model.config)
    inferer = Inferer(config, model_dir=path)
    preprocessor = Preprocessor(config)

    test_example_lambda = lambda example, dataset_name, db_type: test_example(
        model,
        config,
        inferer,
        preprocessor,
        example,
        path,
        device,
        dataset_name,
        db_type,
    )

    return model, inferer, preprocessor, config, test_example_lambda


def test_example(
    model,
    config,
    inferer,
    preprocessor,
    example,
    dir_path,
    device,
    dataset_name="w3schools_test",
    db_type="sqlite",
):
    # Modify test example
    modify_file(config, example.question, example.db_id, db_type=db_type)

    # Preprocess
    # TODO: at the first execution, it takes too long. fix it.
    start_time = time.time()
    block_name = "preprocess"
    logging.info(f"{block_name}...")
    preprocessor.preprocess(is_inference=True)
    logging.info(f"{block_name}: {(time.time() - start_time):.2f}")

    # Infer
    if inferer is None:
        inferer = Inferer(config)
    infer_args = InferConfig(mode="infer", device=device)
    # debug_args = InferConfig(mode='debug')

    dataset_schema = registry.construct("dataset", config["data"]["val"])[0].schema

    # Change output path
    log_path = f"{dir_path}/ie_dirs"
    if not os.path.isdir(log_path):
        os.makedirs(log_path)
    infer_output_path = os.path.join(log_path, "test-infer.jsonl")
    # debug_output_path = os.path.join(log_path, "test-debug.jsonl")

    # Run inferer mode=infer
    start_time = time.time()
    block_name = "inferer"
    logging.info(f"{block_name}...")
    inferer.infer(model, infer_output_path, infer_args)
    logging.info(f"{block_name}: {(time.time() - start_time):.2f}")

    inferred_line = list(open(infer_output_path))[0]
    inferred_result = json.loads(inferred_line)
    assert "beams" in inferred_result
    inferred_sql = inferred_result["beams"][0]["inferred_code"]
    inferred_schema = inferred_result["schema"]
    inferred_sql_dict = inferred_result["beams"][0]["model_output"]
    inferred_confidence = inferred_result["beams"][0]["confidence"]

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

    inferred_sql_dict = infer_from_clause(
        inferred_sql_dict,
        dataset_schema,
        grammar=config["model"]["decoder_preproc"]["grammar"]["name"],
    )

    return inferred_sql, inferred_schema, inferred_sql_dict, inferred_confidence


def infer_value_from_question(question, table, column, db, infer_value_cnt):
    def increase_value_cnt():
        infer_value_cnt[0] += 1

    def is_int(word):
        try:
            int(word)
            return True
        except:
            return False

    def sent_to_words(sent):
        words = sent.replace(".", "").replace("  ", " ").split(" ")
        return words

    pg_config = "host=localhost port=5435 user=postgres password=postgres dbname=" + db
    # Try to find number values from string
    words = sent_to_words(question)
    values = [word for word in words if is_int(word)]
    if values:
        if len(values) <= infer_value_cnt[0]:
            return values[-1]
        else:
            tmp = infer_value_cnt[0]
            increase_value_cnt()
            return values[tmp]
    # Find value from DB (For string values)
    with psycopg2.connect(pg_config) as conn:
        with conn.cursor() as cursor:
            search_query = f"SELECT {column} FROM {table}"
            cursor.execute(search_query)
            results = cursor.fetchall()
            if not results:
                return "value_not_found"
            for result in results:
                if str(result[0]) in question:
                    return result[0]

            return result[0][0]


def build_where_tree(
    cond, tables, columns, col_to_tab, db, infer_value=False, question=None
):
    if cond["_type"] in ["And", "Or"]:
        left_cond = cond["left"]
        left_cond_list = build_where_tree(
            left_cond, tables, columns, col_to_tab, db, infer_value=infer_value
        )

        right_cond = cond["right"]
        right_cond_list = build_where_tree(
            right_cond, tables, columns, col_to_tab, db, infer_value=infer_value
        )

        return [left_cond_list, cond["_type"].upper(), right_cond_list]

    elif cond["_type"] in OPNAME_TO_OP.keys():
        col_id = cond["val_unit"]["col_unit1"]["col_id"]
        col_name = columns[col_id]
        tab_name = tables[col_to_tab[col_id]]
        value = cond["val1"]["_type"]
        if value == "Terminal" and infer_value:
            assert question is not None
            value = infer_value_from_question(question, tab_name, col_name, db)
        formatted_cond = [
            tab_name + "." + col_name,
            OPNAME_TO_OP[cond["_type"]],
            "'" + value + "'",
        ]

        return formatted_cond


def build_where_list_only_and(
    cond,
    tables,
    columns,
    col_to_tab,
    db,
    infer_value=False,
    infer_value_cnt=[0],
    question=None,
    grammar="spider",
):
    if cond["_type"] in ["And", "Or"]:
        left_cond = cond["left"]
        left_cond_list, left_value_list = build_where_list_only_and(
            left_cond,
            tables,
            columns,
            col_to_tab,
            db,
            infer_value=infer_value,
            infer_value_cnt=infer_value_cnt,
        )

        right_cond = cond["right"]
        right_cond_list, right_value_list = build_where_list_only_and(
            right_cond,
            tables,
            columns,
            col_to_tab,
            db,
            infer_value=infer_value,
            infer_value_cnt=infer_value_cnt,
        )

        return left_cond_list + right_cond_list, left_value_list + right_value_list

    elif cond["_type"] in OPNAME_TO_OP.keys():
        if grammar in {"spider"}:
            col_id = cond["val_unit"]["col_unit1"]["col_id"]
        elif grammar in {"postgres", "postgresql"}:
            # 일단 where절에는 concat된 컬럼이 들어오지 않는다고 가정하고, 리스트 길이가 1이라고 하자.
            col_id = cond["val_unit"]["col_unit1"]["concat_columns"]["col_id"][0]
        col_name = columns[col_id]
        tab_name = tables[col_to_tab[col_id]]

        value = cond["val1"]["_type"]
        if value == "Terminal" and infer_value:
            assert question is not None
            value = infer_value_from_question(
                question, tab_name, col_name, db, infer_value_cnt
            )
        if isinstance(value, str):
            formatted_cond = [
                tab_name + "." + col_name,
                OPNAME_TO_OP[cond["_type"]],
                "'" + value + "'",
            ]
        else:
            formatted_cond = [
                tab_name + "." + col_name,
                OPNAME_TO_OP[cond["_type"]],
                str(value),
            ]

        return [formatted_cond], [value]


def build_join_cond_list_only_and(cond, tables, columns, col_to_tab, grammar="spider"):
    if cond["_type"] in ["And", "Or"]:
        left_cond = cond["left"]
        left_cond_list = build_join_cond_list_only_and(
            left_cond, tables, columns, col_to_tab
        )

        right_cond = cond["right"]
        right_cond_list = build_join_cond_list_only_and(
            right_cond, tables, columns, col_to_tab
        )

        return left_cond_list + right_cond_list

    elif cond["_type"] in OPNAME_TO_OP.keys():
        if grammar in {"spider"}:
            col_id1 = cond["val_unit"]["col_unit1"]["col_id"]
            tab_name1 = tables[col_to_tab[col_id1]]
            col_name1 = columns[col_id1]

            col_id2 = cond["val1"]["c"]["col_id"]
            col_name2 = columns[col_id2]
            tab_name2 = tables[col_to_tab[col_id2]]

            formatted_cond = [
                tab_name1
                + "."
                + col_name1
                + OPNAME_TO_OP[cond["_type"]]
                + tab_name2
                + "."
                + col_name2
            ]

        elif grammar in {"postgres", "postgresql"}:
            col_ids1 = list(
                map(lambda x: x["col_id"], cond["val1"]["c"]["concat_columns"])
            )
            col_names1 = list(
                map(lambda x: f"{tables[col_to_tab[x]]}.{columns[x]}", col_ids1)
            )
            col_ids2 = list(
                map(lambda x: x["col_id"], cond["val1"]["c"]["concat_columns"])
            )
            col_names2 = list(
                map(lambda x: f"{tables[col_to_tab[x]]}.{columns[x]}", col_ids2)
            )

            formatted_cond = [
                ' ||"_"|| '.join(col_names1)
                + OPNAME_TO_OP[cond["_type"]]
                + ' ||"_"|| '.join(col_names2)
            ]
        else:
            raise KeyError(f"no such grammar: {grammar}")

        return [formatted_cond]


def infer_from_clause(sql_dict, schema, grammar="postgres"):
    if grammar in {"postgres", "postgresql"}:
        asdl_file = "Postgres.asdl"
    elif grammar in {"spider"}:
        asdl_file = "Spider_f2.asdl"
    else:
        raise KeyError(f"wrong grammar name: {grammar}")
    custom_primitive_type_checkers = {}
    custom_primitive_type_checkers["table"] = lambda x: isinstance(x, int)
    custom_primitive_type_checkers["column"] = lambda x: isinstance(x, int)

    ast_wrapper = ast_util.ASTWrapper(
        asdl.parse(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "ratsql",
                "grammars",
                asdl_file,
            )
        ),
        custom_primitive_type_checkers=custom_primitive_type_checkers,
    )

    if grammar in {"postgres", "postgresql"}:
        unparser = PostgresUnparser(ast_wrapper, schema, 2, do_refine_from=True)
    elif grammar in {"spider"}:
        unparser = SpiderUnparser(ast_wrapper, schema, 2, do_refine_from=True)
    else:
        raise KeyError(f"wrong grammar name: {grammar}")
    unparser.refine_from(sql_dict)

    return sql_dict


# Jhcho - 21.11.05: Fix schema issue
def build_schema_dict(dataset_name):
    global DB_SCHEMA
    tables_file = json.load(open(os.path.join("data", dataset_name, "tables.json")))
    for ent in tables_file:
        db_id = ent["db_id"]
        table_names = ent["table_names_original"]
        column_names = []
        col_to_tab = {}
        for col_id, (tab_id, col_name) in enumerate(ent["column_names_original"]):
            column_names.append(col_name)
            col_to_tab[col_id] = tab_id
        DB_SCHEMA[db_id] = {
            "tables": table_names,
            "columns": column_names,
            "col_to_tab": col_to_tab,
        }


def is_number(text):
    try:
        float(text)
        return True
    except:
        return False

def findnth(haystack, needle, n):
    parts= haystack.split(needle, n+1)
    if len(parts)<=n+1:
        return -1
    return len(haystack)-len(parts[-1])-len(needle)


def format_result(
    sql,
    sql_dict,
    db_id,
    confidence=0.0,
    pred_query_type="INFORM_SQL",
    infer_value=False,
    question=None,
    schema=None,
    grammar="spider",
    infer_value_cnt=[0],
):
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
    tables = DB_SCHEMA[db_id]["tables"]
    columns = DB_SCHEMA[db_id]["columns"]
    col_to_tab = DB_SCHEMA[db_id]["col_to_tab"]

    schema_dict = {k: [] for k in tables}
    for col_id, col_name in enumerate(columns):
        tab_id = col_to_tab[col_id]
        if not tab_id:
            continue
        tab_name = tables[tab_id]
        schema_dict[tab_name].append(col_name)

    select_list = []
    for agg in sql_dict["select"]["aggs"]:
        agg_type = agg["agg_id"]["_type"]
        if agg_type == "NoneAggOp":
            agg_name = "NONE"
        else:
            agg_name = agg_type.upper()

        if grammar in {"spider"}:
            col_id = agg["val_unit"]["col_unit1"]["col_id"]
            tab_name = tables[col_to_tab[col_id]]
            col_name = columns[col_id]
            select_list.append([agg_name, tab_name + "." + col_name])
        elif grammar in {"postgres", "postgresql"}:
            col_ids = list(
                map(
                    lambda x: x["col_id"],
                    agg["val_unit"]["col_unit1"]["concat_columns"],
                )
            )
            col_names = list(
                map(
                    lambda x: f"{tables[col_to_tab[x]]}.{columns[x]}"
                    if "*" not in columns[x]
                    else "*",
                    col_ids,
                )
            )
            select_list.append([agg_name, ' ||"_"|| '.join(col_names)])
        else:
            raise KeyError(f"no such grammar: {grammar}")

    from_list = []
    for tab in sql_dict["from"]["table_units"]:
        tab_id = tab["table_id"]
        tab_name = tables[tab_id]
        from_list.append(tab_name)

    join_cond_list = []
    if "conds" in sql_dict["from"]:
        join_cond = sql_dict["from"]["conds"]
        # join_cond_list = build_where_tree(join_cond, tables, columns, col_to_tab)
        join_cond_list = build_join_cond_list_only_and(
            join_cond, tables, columns, col_to_tab, grammar=grammar
        )  # Accept only and conditions
    
    if 'except' in sql_dict['sql_ieu']:
        set_keyword = 'except'
    elif 'intersect' in sql_dict['sql_ieu']:
        set_keyword = 'intersect'
    elif 'union' in sql_dict['sql_ieu']:
        set_keyword = 'union'
    else:
        set_keyword = None
    
    join_cond2 = []
    join_cond_list2 = []
    
    if set_keyword is not None:
        if "conds" in sql_dict['sql_ieu'][set_keyword]["from"]:
            join_cond2 = sql_dict['sql_ieu'][set_keyword]["from"]["conds"]
            join_cond_list2 = build_join_cond_list_only_and(join_cond2, tables, columns, col_to_tab, grammar=grammar) 
        

    where_list = []
    value_list = []
    try:
        if "where" in sql_dict["sql_where"]:
            where_cond = sql_dict["sql_where"]["where"]
            # where_list = build_where_tree(where_cond, tables, columns, col_to_tab)
            where_list, value_list = build_where_list_only_and(
                where_cond,
                tables,
                columns,
                col_to_tab,
                db_id,
                infer_value=infer_value,
                infer_value_cnt=infer_value_cnt,
                question=question,
            )  # Accept only and conditions
    except:
        pass

    # for value in value_list:
    #     sql = sql.replace("'terminal'", value, 1)

    group_list = []
    if "group_by" in sql_dict["sql_groupby"]:
        for group in sql_dict["sql_groupby"]["group_by"]:
            if grammar in {"spider"}:
                col_id = group["col_id"]
                tab_name = tables[col_to_tab[col_id]]
                col_name = columns[col_id]
                group_list.append(tab_name + "." + col_name)
            elif grammar in {"postgres", "postgresql"}:
                col_ids = list(map(lambda x: x["col_id"], group["concat_columns"]))
                col_names = list(
                    map(lambda x: f"{tables[col_to_tab[x]]}.{columns[x]}", col_ids)
                )
                group_list.append([agg_name, ' ||"_"|| '.join(col_names)])
            else:
                raise KeyError(f"no such grammar: {grammar}")

    value_list = []
    try:
        for key in ["intersect", "union", "except"]:
            if key in sql_dict["sql_ieu"]:
                if "where" in sql_dict["sql_ieu"][key]["sql_where"]:
                    where_list, value_list = build_where_list_only_and(
                        sql_dict["sql_ieu"][key]["sql_where"]["where"],
                        tables,
                        columns,
                        col_to_tab,
                        db_id,
                        infer_value=infer_value,
                        infer_value_cnt=infer_value_cnt,
                        question=question,
                    )  # Accept only and conditions
    except:
        pass

    # for value in value_list:
    #     sql = sql.replace("'terminal'", value, 1)

    # Create FROM Clause (to support POSTGRESQL)
    if join_cond_list:
        new_from_clause_str = " FROM "
        seen_tables = []
        for join_cond in join_cond_list:
            assert len(join_cond) == 1
            join_cond = join_cond[0]
            # Get table names
            t1, t2 = join_cond.split('=')
            t1 = t1.split('.')[0]
            t2 = t2.split('.')[0]
            if t1 not in seen_tables and t2 not in seen_tables: 
                new_from_clause_str += f"{t1} JOIN {t2} ON {join_cond} "
            elif t1 not in seen_tables:
                new_from_clause_str += f"JOIN {t1} ON {join_cond} "
            elif t2 not in seen_tables:
                new_from_clause_str += f"JOIN {t2} ON {join_cond} "
            else:
                raise RuntimeError("Should not be here... something wrong")
            seen_tables.append(t1)
            seen_tables.append(t2)
        # Replace from clause
        start_idx = sql.index(" FROM ")
        possible_end_indices = []
        for keyword in [" WHERE ", " GROUP BY ", " ORDER BY ", " HAVING "]:
            if keyword in sql:
                possible_end_indices.append(sql.index(keyword))
        if possible_end_indices:
            end_idx = min(possible_end_indices)
        else:
            end_idx = len(sql)
        sql = sql[:start_idx] + new_from_clause_str + sql[end_idx:]    
    if join_cond_list2:
        new_from_clause_str = " FROM "
        seen_tables = []
        for join_cond in join_cond_list2:
            assert len(join_cond) == 1
            join_cond = join_cond[0]
            # Get table names
            t1, t2 = join_cond.split('=')
            t1 = t1.split('.')[0]
            t2 = t2.split('.')[0]
            if t1 not in seen_tables and t2 not in seen_tables: 
                new_from_clause_str += f"{t1} JOIN {t2} ON {join_cond} "
            elif t1 not in seen_tables:
                new_from_clause_str += f"JOIN {t1} ON {join_cond} "
            elif t2 not in seen_tables:
                new_from_clause_str += f"JOIN {t2} ON {join_cond} "
            else:
                raise RuntimeError("Should not be here... something wrong")
            seen_tables.append(t1)
            seen_tables.append(t2)
        # Replace from clause
        sql_right = sql.split(f" {set_keyword.upper()} ")[1]
        start_idx = sql_right.index(" FROM ")
        possible_end_indices = []
        for keyword in [" WHERE ", " GROUP BY ", " ORDER BY ", " HAVING "]:
            if keyword in sql_right:
                possible_end_indices.append(sql_right.index(keyword))
        if possible_end_indices:
            end_idx = min(possible_end_indices)
        else:
            end_idx = len(sql_right)
        sql_right = sql_right[:start_idx] + new_from_clause_str + sql_right[end_idx:]
        sql = sql.split(f" {set_keyword.upper()} ")[0] + f" {set_keyword.upper()} " + sql_right
    

    # Replace value
    value_dic = ["TV Lounge", "Study Room", "Super bootcamp", "Auditions"]
    value_found = []
    questions = question.split("<s>")[1:]
    if 'INTERSECT' in sql or 'EXCEPT' in sql:
        inc_num = 100
    else:
        inc_num = -100
    for idx, single_question in enumerate(questions):
        mul_idx = idx if idx else len(questions)
        for value in value_dic:
            if value in single_question:
                value_found.append((value, single_question.index(value) + mul_idx * inc_num))
        for tok in single_question.split(' '):
            if is_number(tok):
                value_found.append((tok, single_question.index(tok) + mul_idx * inc_num))
        
    if value_found:
        # Sort
        value_found = sorted(value_found, key=lambda x: x[1])
        # Remove repetition
        filtered_value_found = []
        for value in value_found:
            if value[0] not in filtered_value_found:
                filtered_value_found.append(value[0])
        value_found = filtered_value_found

    sql_toks = []
    sql_split = sql.split(' ')
    for idx, sql_tok in enumerate(sql_split):
        if "'terminal'" in sql_tok and value_found:
            target_table, target_column = sql_split[idx-2].split('.')
            target_type = [col.type for col in schema.columns if col.orig_name == target_column][0]
            selected_value = None
            if target_type == 'number':
                for val_idx, value in enumerate(value_found):
                    if is_number(value):
                        selected_value = value_found.pop(val_idx)
                        break
            elif target_type == 'text':
                for val_idx, value in enumerate(value_found):
                    if not is_number(value):
                        selected_value = value_found.pop(val_idx)
                        break
            if sql_tok[-1] == ')':
                sql_tok = f"'{selected_value}')"
            else:
                sql_tok = f"'{selected_value}'"
        sql_toks.append(sql_tok)
    sql = ' '.join(sql_toks)

    formatted_sql_dict = {
        "select": select_list,
        "from": from_list,
        "where": where_list,
        "groupby": group_list,
        "join_conditions": join_cond_list,
    }

    result = (
        json.dumps(formatted_sql_dict)
        + "\n"
        + db_id
        + "\n"
        + sql
        + "\n"
        + json.dumps(schema_dict)
        + "\n"
        + f"{confidence*100:.2f}"
        + "\n"
        + pred_query_type
    )
    logging.info(f"Final:{sql} ({pred_query_type})({confidence*100:.2f})")
    return result


def format_result_scan_table(db_id, question):

    global DB_SCHEMA
    tables = DB_SCHEMA[db_id]["tables"]
    columns = DB_SCHEMA[db_id]["columns"]
    col_to_tab = DB_SCHEMA[db_id]["col_to_tab"]

    indicated_table = tables[0]
    for table in tables:
        if table in question:
            indicated_table = table

    schema_dict = {k: [] for k in tables}
    columns_in_indicated_table = []

    for col_id, col_name in enumerate(columns):
        tab_id = col_to_tab[col_id]
        if not tab_id:
            continue
        tab_name = tables[tab_id]
        schema_dict[tab_name].append(col_name)
        if tab_name == indicated_table:
            columns_in_indicated_table.append(col_name)

    select_list = []
    for col_name in columns_in_indicated_table:
        select_list.append(["NONE", indicated_table + "." + col_name])

    formatted_sql_dict = {
        "select": select_list,
        "from": [indicated_table],
        "where": [],
        "groupby": [],
        "join_conditions": [],
    }

    result = (
        json.dumps(formatted_sql_dict) + "\n" + db_id + "\n" + json.dumps(schema_dict)
    )

    return result


def rule_based_nl2sql(db, tables, question, grammar, dataset_name):
    # if the question doesn't satisfy predefined rules, return none
    def match_rules(question):
        if "테이블 전체 조회" in question:
            return 1
        else:
            return 0

    rule = match_rules(question)
    if not rule:
        return None, None, None

    # according to the satisfied rules, generate sql
    def gen_sql_by_rule(rule, question):
        if rule == 1:
            table = question[0]
            return f"select * from {table}"
        else:
            raise KeyError(f"no such rule: {rule}")

    sql = gen_sql_by_rule(rule, question)

    # fill sql_dict
    _schema = Schema(get_schema(db, db_type, tables))
    if grammar in {"postgres", "postgresql"}:
        raise RuntimeError("Unsupported grammar: {grammar}")
        # sql_dict = PostgresLanguage().parse(get_sql(_schema, sql), "train")
    elif grammar in {"spider"}:
        sql_dict = SpiderLanguage().parse(get_sql(_schema, sql), "train")
    else:
        raise KeyError(f"no such grammar: {grammar}")

    # fill schema
    example = Example(question=question, gold=sql, db_id=db, dataset_name=dataset_name)

    return sql, schema, sql_dict


def isHanguel(text):
    hanCount = len(re.findall("[\u3130-\u318F\uAC00-\uD7A3]+", text))
    return hanCount > 0


def main(args):
    eng_model_path = args.eng_model_path
    kor_model_path = args.kor_model_path
    assert kor_model_path or eng_model_path
    db_type = args.db_type
    dataset_name = args.dataset_name
    mode = args.mode
    server_ip = args.server_ip
    server_port = args.server_port
    path_dic = {
        "config": "/mnt/sdb1/shpark/logdir/cosql-history/bs=24,lr=7.4e-04,bert_lr=3.0e-06,end_lr=0e0,seed=4/config-20231101T093016.json",
        "model" : "/mnt/sdb1/shpark/logdir/cosql-history/bs=24,lr=7.4e-04,bert_lr=3.0e-06,end_lr=0e0,seed=4/model_checkpoint-013000.pt",
        "output": "/mnt/sdb1/shpark/logdir/cosql-history/bs=24,lr=7.4e-04,bert_lr=3.0e-06,end_lr=0e0,seed=4/possible_outputs.json",
    }
    analysis = args.analysis
    kor_device = "cuda:0"
    eng_device = "cuda:0"
    user_intent_device = eng_device
    db_path = "/home/nl2qgm/data/spider/database/"
    table_path = "/home/nl2qgm/data/spider/tables.json"

    model_load_start_time = time.time()
    if kor_model_path:
        kor_model, kor_preprocessor, model_captum, analysis_config = get_model_from_path2(kor_model_path, kor_device, db_path, table_path)
        
    if eng_model_path:
        eng_model, eng_preprocessor, model_captum, analysis_config = get_model_from_path2(eng_model_path, eng_device, db_path, table_path)
        #user_intent_model = get_user_intent_model(path_dic, user_intent_device, eng_preprocessor)

    logging.info(f"Model Load Time: {(time.time() - model_load_start_time):.2f}")

    build_schema_dict(dataset_name)

    if mode == "server":
        serverSock = socket(AF_INET, SOCK_STREAM)
        server_conn_flag = False
        logging.info(
            f"Waiting for the socket to be connected... (ip:{server_ip}, port:{server_port})"
        )
        while not server_conn_flag:
            try:
                serverSock.bind((server_ip, server_port))
                server_conn_flag = True
            except:
                time.sleep(0.5)
        logging.info("Socket connected! Ready for getting input")

    try:
        while 1:
            try:
                if mode == "server":
                    serverSock.listen(1)
                    connectionSock, addr = serverSock.accept()
                    data = connectionSock.recv(1024).decode("utf-8")
                    if data == "exit":
                        serverSock.close()
                        break
                    db_id, question = data.split("|||")
                elif mode == "user_input":
                    if args.db_id:
                        db_id = args.db_id
                    else:
                        db_id = input("db_id: ")
                    question = input("question: ")
                if question == 'exit':
                    break
                question = question.strip(' ')
                logging.info("Data received")

                infer_start_time = time.time()
                
                if (not eng_model_path and kor_model_path) or (isHanguel(question) and kor_model_path and eng_model_path):
                    logging.info("Use Korean Model")
                    model = kor_model
                    preprocessor = kor_preprocessor    
                else:
                    logging.info("Use English Model")
                    model = eng_model
                    preprocessor = eng_preprocessor
                
                # Infer
                orig_item, preproc_item = preprocessor.run(question, db_id)
                beams = spider_beam_search.beam_search_with_heuristics(model, orig_item, (preproc_item, None), beam_size=2, max_steps=150)
                confidence = torch.softmax(torch.tensor([tmp.score for tmp in beams]), dim=0).cpu().numpy()[0]
                sql_dict, inferred_code = beams[0].inference_state.finalize()
                sql = inferred_code
                if analysis:
                    attributor = Attribution(analysis_config)
                    html = attributor.run(model_captum, question, orig_item, preproc_item)
                # User intent
                #question_changed = user_intent_model.change_input_format(question)
                orig_item, preproc_item = eng_preprocessor.run(question, db_id)
                #orig_item, preproc_item = eng_preprocessor.run(question_changed, db_id)
                #pred_label = user_intent_model([preproc_item])
                pred_label = "INFORM_SQL"
                logging.info("Inference Time: %f\n\n" % (time.time() - infer_start_time))
                logging.info(question)
                logging.info(sql)
                logging.info(pred_label)

                if mode == "server":
                    final_result = format_result(
                        sql,
                        sql_dict,
                        db_id,
                        confidence,
                        pred_label,
                        infer_value=True,
                        question=question,
                        schema=preprocessor.schemas[db_id],
                    )
                    logging.info(final_result)
                    connectionSock.send(final_result.encode("utf-8"))
                    connectionSock.close()
                else:
                    final_result = format_result(
                        sql,
                        sql_dict,
                        db_id,
                        confidence,
                        pred_label,
                        infer_value=True,
                        question=question,
                        schema=preprocessor.schemas[db_id],
                    )
            except:
                if mode == "server":
                    connectionSock.send("".encode("utf-8"))
                    connectionSock.close()
                traceback.print_exc()

    except KeyboardInterrupt:
        try:
            if mode == "server":
                serverSock.close()
        except:
            pass


if __name__ == "__main__":
    def _get_arguments():
        parser = argparse.ArgumentParser()

        model_path_group = parser.add_argument_group("model_path")
        model_path_group.add_argument("--eng_model_path", type=str, default="saved_model/eng/")
        model_path_group.add_argument("--kor_model_path", type=str)

        db_group = parser.add_argument_group("db")
        db_group.add_argument("--db_id", type=str, default="concert_singer")
        db_group.add_argument("--db_type", type=str, default="postgres")

        server_group = parser.add_argument_group("server")
        server_group.add_argument("--server_ip", type=str, default="localhost")
        server_group.add_argument("--server_port", type=int, default=8290)

        parser.add_argument("--dataset_name", type=str, default="spider")
        parser.add_argument("--mode", type=str, choices=["server", "user_input"], default="user_input")
        parser.add_argument("--analysis", type=bool, default=True)

        args = parser.parse_args()
        if not (args.eng_model_path or args.kor_model_path):
            parser.error("add at least one model_path")

        return args

    args = _get_arguments()

    logging.basicConfig(level=logging.INFO)
    main(args)