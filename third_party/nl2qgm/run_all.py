import os
import attr
import tqdm
import json
import torch
import argparse
import _jsonnet

from nltk.tokenize import word_tokenize

# These imports are needed for registry.lookup
# noinspection PyUnresolvedReferences
from ratsql import datasets
# noinspection PyUnresolvedReferences
from ratsql import grammars
# noinspection PyUnresolvedReferences
from ratsql import models
# noinspection PyUnresolvedReferences
from ratsql.utils import registry
# noinspection PyUnresolvedReferences
from ratsql.utils import vocab, evaluation
from ratsql.commands.infer import Inferer
from ratsql.datasets.spider_lib.process_sql import get_sql, get_schema, Schema

from run import InferConfig, EvalConfig

PROJECT_DIR = "/home/hkkang/NL2QGM"
BERT_CONFIG_PATH = os.path.join(PROJECT_DIR, "experiments/spider-bert-run-no-join.jsonnet")
ELECTRA_CONFIG_PATH = os.path.join(PROJECT_DIR, "experiments/spider-electra-run-no-join.jsonnet")
GLOVE_CONFIG_PATH = os.path.join(PROJECT_DIR, "experiments/spider-glove-run-no-join.jsonnet")

def is_int(text):
    try:
        int(text)
        return True
    except:
        return False

@attr.s
class TestInfo:
    question = attr.ib()
    gold = attr.ib()
    db_id = attr.ib()
    model_seed = attr.ib()
    model_type= attr.ib(default='bert')
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


class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.model_preproc = registry.instantiate(
            registry.lookup('model', config['model']).Preproc,
            config['model'])

    def preprocess(self, manual_linking_info=None, is_inference=False):
        self.model_preproc.clear_items()
        # Create data info dic:
        for section, config in self.config['data'].items():
            data = registry.construct('dataset', config)
            for idx, item in enumerate(tqdm.tqdm(data, desc=f"{section} section", dynamic_ncols=True)):
                to_add, validation_info = self.model_preproc.validate_item(item, section)
                if to_add or is_inference:
                    self.model_preproc.add_item(item, section, validation_info, manual_linking_info)
                else:
                    print(f"Skipping... section:{section} idx:{idx}")
                    to_add, validation_info = self.model_preproc.validate_item(item, section)
        self.model_preproc.save(is_inference=True)

def modify_file(question, db_id, gold=None):
    file_path = os.path.join(PROJECT_DIR, "data/spider_test/test.json")
    table_path = os.path.join(PROJECT_DIR, "data/spider_test/tables.json")
    db_dir = os.path.join(PROJECT_DIR, "data/spider_test/database/")

    # Check valid db_id
    with open(table_path, 'r') as f:
        table_data = json.load(f)
    dbs = {item['db_id']: item for item in table_data}
    assert db_id in dbs, f"{db_id} is not a valid db_id"
    # Read in data file
    with open(file_path, 'r') as f:
        data = json.load(f)
    # Modify data
    data[0]['question'] = question
    data[0]['question_toks'] = word_tokenize(question)
    data[0]['db_id'] = db_id
    if gold:
        data[0]['query'] = gold
        data[0]['query_toks'] = word_tokenize(gold)
    # Parse SQL into spider json format
    db_path = os.path.join(db_dir, db_id, db_id + '.sqlite')    
    assert os.path.isfile(db_path), f'{db_path} does not exist!'
    schema = Schema(get_schema(db_path), dbs[db_id])
    g_sql = get_sql(schema, gold)
    data[0]['sql'] = g_sql
    # Overwrite data file
    with open(file_path, 'w') as f:
        f.write(json.dumps(data, indent=4))
    return None


def get_preproc_configs(seed, model_type='bert'):
    if model_type == 'bert':
        CONFIG_PATH = BERT_CONFIG_PATH
    elif model_type == 'electra':
        CONFIG_PATH = ELECTRA_CONFIG_PATH
    elif model_type == 'glove':
        CONFIG_PATH = GLOVE_CONFIG_PATH
    elif model_type == 't5':
        CONFIG_PATH = ELECTRA_CONFIG_PATH
    else:
        raise NotImplementedError
    # Preprocess
    ## Load Preprocess Model
    exp_config = json.loads(_jsonnet.evaluate_file(CONFIG_PATH))
    model_config_file = exp_config["model_config"]
    model_config_args = exp_config["model_config_args"]
    ## Change seed and set logdir
    exp_config['model_config_args']['seed'] = seed
    assert is_int(exp_config['logdir'].split('_')[-1]), f"is not int: {exp_config['logdir'].split('_')[-1]}"
    logdir = '_'.join(exp_config['logdir'].split('_')[:-1] + [str(seed)])
    exp_config['logdir'] = logdir

    config = json.loads(_jsonnet.evaluate_file(model_config_file, tla_codes={'args': json.dumps(model_config_args)}))
    
    # Change data
    eval_section = 'test'
    del config['data']['train']
    del config['data']['val']
    config['data'][eval_section] = {
            'db_path': 'data/spider_test/database',
            'name': 'spider',
            'paths': ['data/spider_test/test.json'],
            'tables_paths': ['data/spider_test/tables.json'],
    }
    config['model']['decoder_preproc']['save_path'] = config['model']['decoder_preproc']['save_path'].replace('spider', 'spider_test')
    config['model']['encoder_preproc']['save_path'] = config['model']['encoder_preproc']['save_path'].replace('spider', 'spider_test')
    config['model']['encoder_preproc']['db_path'] = config['model']['encoder_preproc']['db_path'].replace('spider', 'spider_test')
    
    return config, exp_config


def get_infer_args(seed, model_name, model_type='bert', step=None):
    # Preprocess
    ## Load Preprocess Model
    if model_type == 'bert':
        CONFIG_PATH = BERT_CONFIG_PATH
    elif model_type == 'electra':
        CONFIG_PATH = ELECTRA_CONFIG_PATH
    elif model_type == 'glove':
        CONFIG_PATH = GLOVE_CONFIG_PATH
    elif model_type == 't5':
        CONFIG_PATH = ELECTRA_CONFIG_PATH
    else:
        raise NotImplementedError
    exp_config = json.loads(_jsonnet.evaluate_file(CONFIG_PATH))

    ## Change seed and set logdir
    exp_config['model_config_args']['seed'] = seed
    assert is_int(exp_config['logdir'].split('_')[-1])
    logdir = '_'.join(exp_config['logdir'].split('_')[:-1] + [str(seed)])
    exp_config['logdir'] = logdir

    # Infer
    mode = 'infer'
    if step is None:
        step = exp_config['eval_steps'][0]
    infer_output_path = f"{exp_config['eval_output']}/{exp_config['eval_name']}-step_{step}-{mode}.jsonl"
    args = InferConfig(
        exp_config["model_config"],
        json.dumps(exp_config["model_config_args"]),
        exp_config['logdir'],
        "test",
        exp_config['eval_beam_size'],
        infer_output_path,
        step,
        mode=mode,
        use_heuristic=exp_config['eval_use_heuristic'],
    )
    args.logdir = os.path.join(args.logdir, model_name)
    return args


def load_model(seed, model_type='bert'):
    config, exp_config = get_preproc_configs(seed, model_type=model_type)
    args = get_infer_args(seed, config['model_name'], model_type=model_type)
    inferer = Inferer(config)
    return inferer.load_model(args.logdir)


def test_example(model, test_info: TestInfo, step=None):
    seed = test_info.model_seed
    
    # Modify test set
    modify_file(test_info.question, test_info.db_id, test_info.gold)

    # Get basic config 
    config, exp_config = get_preproc_configs(seed, model_type=test_info.model_type)
    
    # Preprocess
    preprocessor = Preprocessor(config)
    preprocessor.preprocess(test_info.manual_linking_info)

    # Inferer
    args = get_infer_args(seed, config['model_name'], model_type=test_info.model_type, step=step)
    inferer = Inferer(config)

    # Change output path
    infer_output_path = args.output.replace('__LOGDIR__', args.logdir)
    infer_output_path = infer_output_path.replace('.jsonl', '-testing.jsonl')

    # Run inferer mode=infer
    inferer.infer(model, infer_output_path, args)

    # Run inferer mode=debug
    args.mode = 'debug'
    args.output = args.output.replace('infer', 'debug')
    debug_output_path = infer_output_path.replace('infer', 'debug')
    inferer.infer(model, debug_output_path, args)

    # Eval
    eval_output_path = debug_output_path.replace('debug', 'eval').replace('.jsonl', '.json')
    eval_config = EvalConfig(
        exp_config["model_config"],
        json.dumps(exp_config["model_config_args"]),
        exp_config['logdir'],
        "test",
        infer_output_path,
        eval_output_path,
        False,
    )

    section = eval_config.section
    inferred_path = eval_config.inferred

    inferred = open(inferred_path)
    data = registry.construct('dataset', config['data'][section])
    metrics = data.Metrics(data, config['data'][section]['tables_paths'][0])

    inferred_lines = list(inferred)
    if len(inferred_lines) < len(data):
        data = evaluation.filter_examples(inferred_lines, data)

    for line in inferred_lines:
        infer_results = json.loads(line)
        if infer_results['beams']:
            inferred_code = infer_results['beams'][0]['inferred_code']
        else:
            inferred_code = None
        if 'index' in infer_results:
            result = metrics.add(data[infer_results['index']], inferred_code)
        else:
            metrics.add(None, inferred_code, obsolete_gold_code=infer_results['gold_code'])

    with open(eval_output_path, 'w') as f:
        f.write(json.dumps(metrics.finalize(), indent=4))
    
    print(f'Wrote eval results to {eval_output_path}')


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('question', type=str)
    parser.add_argument('gold', type=str)
    parser.add_argument('db_id', type=str, help='name of DB')
    parser.add_argument('model_seed', type=int, help='seed that the model is trained on')
    parser.add_argument('model_type', type=str, default='bert', help='bert or electra')
    parser.add_argument('--cem', type=str, nargs='+', default= [], help='indices for manual column exatch match')
    parser.add_argument('--tem', type=str, nargs='+', default= [], help='indices for manual table exatch match')
    parser.add_argument('--cpm', type=str, nargs='+', default= [], help='indices for manual column partial match')
    parser.add_argument('--tpm', type=str, nargs='+', default= [], help='indices for manual table partial match')
    parser.add_argument('--cm', type=str, nargs='+', default= [], help='indices for manual cell match')
    parser.add_argument('--nm', type=str, nargs='+', default= [], help='indices for manual number match')
    parser.add_argument('--dm', type=str, nargs='+', default= [], help='indices for manual date match')
    parser.add_argument('--cem_exclude', type=str, nargs='+', default= [], help='indices for manual column exatch match')
    parser.add_argument('--tem_exclude', type=str, nargs='+', default= [], help='indices for manual table exatch match')
    parser.add_argument('--cpm_exclude', type=str, nargs='+', default= [], help='indices for manual column partial match')
    parser.add_argument('--tpm_exclude', type=str, nargs='+', default= [], help='indices for manual table partial match')
    parser.add_argument('--cm_exclude', type=str, nargs='+', default= [], help='indices for manual cell match')
    parser.add_argument('--nm_exclude', type=str, nargs='+', default= [], help='indices for manual number match')
    parser.add_argument('--dm_exclude', type=str, nargs='+', default= [], help='indices for manual date match')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    test_info = TestInfo(**vars(args))
    
    # Load model
    trained_model, last_step = load_model(test_info.model_seed, test_info.model_type)
    
    test_example(trained_model, test_info, step=last_step)

