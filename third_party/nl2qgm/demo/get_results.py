import json
import os
import random
from typing import *
from get_model import get_model
from utils import add_value_one_sql
from process_sql import tokenize, get_schema, get_tables_with_alias, Schema, get_sql
from ratsql.models.spider import spider_beam_search

# Flag to disable value evaluation
DISABLE_VALUE = True
# Flag to disable distinct in select evaluation
DISABLE_DISTINCT = True


CLAUSE_KEYWORDS = ('select', 'from', 'where', 'group', 'order', 'limit', 'intersect', 'union', 'except')
JOIN_KEYWORDS = ('join', 'on', 'as')

WHERE_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists')
UNIT_OPS = ('none', '-', '+', "*", '/')
AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg')
TABLE_TYPE = {
    'sql': "sql",
    'table_unit': "table_unit",
}

COND_OPS = ('and', 'or')
SQL_OPS = ('intersect', 'union', 'except')
ORDER_OPS = ('desc', 'asc')


HARDNESS = {
    "component1": ('where', 'group', 'order', 'limit', 'join', 'or', 'like'),
    "component2": ('except', 'union', 'intersect')
}

random.seed(0)

file_dir_path = os.path.dirname(os.path.abspath(__file__))  # Intialize config

config_file_path = os.path.join(file_dir_path, "backend_config.json")
with open(config_file_path, "r") as config_file:
    config = json.load(config_file)

text_to_sql_model, text_to_sql_preprocessor = get_model(
    config, model_type="text_to_sql"
)

def text_to_sql(text, db_id):
    text = text
    db_id = db_id
    text_history = ""
    input_text = "<s> " + text
    orig_item, preproc_item = text_to_sql_preprocessor.run(input_text, db_id)
    if not text_history.endswith(text):
        text_history += " <s> " + text
    # translate text to sql
    beams = spider_beam_search.beam_search_with_heuristics(
        text_to_sql_model,
        orig_item,
        (preproc_item, None),
        beam_size=config["model"]["text_to_sql"]["beam_size"],
        max_steps=config["model"]["text_to_sql"]["max_steps"],
    )
    sql_dict, inferred_code = beams[0].inference_state.finalize()
    inferred_code = add_value_one_sql(
        question=text, db_name=db_id, sql=inferred_code, history=text_history
    )
    return inferred_code
def count_component1(sql):
    count = 0
    if len(sql['where']) > 0:
        count += 1
    if len(sql['groupBy']) > 0:
        count += 1
    if len(sql['orderBy']) > 0:
        count += 1
    if sql['limit'] is not None:
        count += 1
    if len(sql['from']['table_units']) > 0:  # JOIN
        count += len(sql['from']['table_units']) - 1

    ao = sql['from']['conds'][1::2] + sql['where'][1::2] + sql['having'][1::2]
    count += len([token for token in ao if token == 'or'])
    cond_units = sql['from']['conds'][::2] + sql['where'][::2] + sql['having'][::2]
    count += len([cond_unit for cond_unit in cond_units if cond_unit[1] == WHERE_OPS.index('like')])

    return count
def has_agg(unit):
    return unit[0] != AGG_OPS.index('none')
def get_nestedSQL(sql):
    nested = []
    for cond_unit in sql['from']['conds'][::2] + sql['where'][::2] + sql['having'][::2]:
        if type(cond_unit[3]) is dict:
            nested.append(cond_unit[3])
        if type(cond_unit[4]) is dict:
            nested.append(cond_unit[4])
    if sql['intersect'] is not None:
        nested.append(sql['intersect'])
    if sql['except'] is not None:
        nested.append(sql['except'])
    if sql['union'] is not None:
        nested.append(sql['union'])
    return nested
def count_component2(sql):
    nested = get_nestedSQL(sql)
    return len(nested)
def count_agg(units):
    return len([unit for unit in units if has_agg(unit)])
def count_others(sql):
    count = 0
    # number of aggregation
    agg_count = count_agg(sql['select'][1])
    agg_count += count_agg(sql['where'][::2])
    agg_count += count_agg(sql['groupBy'])
    if len(sql['orderBy']) > 0:
        agg_count += count_agg([unit[1] for unit in sql['orderBy'][1] if unit[1]] +
                            [unit[2] for unit in sql['orderBy'][1] if unit[2]])
    agg_count += count_agg(sql['having'])
    if agg_count > 1:
        count += 1

    # number of select columns
    if len(sql['select'][1]) > 1:
        count += 1

    # number of where conditions
    if len(sql['where']) > 1:
        count += 1

    # number of group by clauses
    if len(sql['groupBy']) > 1:
        count += 1

    return count

def eval_hardness(sql):
    count_comp1_ = count_component1(sql)
    count_comp2_ = count_component2(sql)
    count_others_ = count_others(sql)

    if count_comp1_ <= 1 and count_others_ == 0 and count_comp2_ == 0:
        return "easy"
    elif (count_others_ <= 2 and count_comp1_ <= 1 and count_comp2_ == 0) or \
            (count_comp1_ <= 2 and count_others_ < 2 and count_comp2_ == 0):
        return "medium"
    elif (count_others_ > 2 and count_comp1_ <= 2 and count_comp2_ == 0) or \
            (2 < count_comp1_ <= 3 and count_others_ <= 2 and count_comp2_ == 0) or \
            (count_comp1_ <= 1 and count_others_ == 0 and count_comp2_ <= 1):
        return "hard"
    else:
        return "extra"

if __name__ == '__main__':
    data_path = "/root/nl2qgm/data/spider/train_spider.json"
    db_path = "/root/nl2qgm/data/spider/database"
    #load instances
    with open(data_path, "r") as data_file:
        data = json.load(data_file)
    responds = {}
    golds = {}
    for id, datum in enumerate(data):
        text = datum["question"]
        db_id = datum["db_id"]
        gold = datum["query"]
        try:
            db_name = db_id
            db = os.path.join(db_path, db_id, db_id + ".sqlite")
            schema = Schema(get_schema(db))
            g_sql = get_sql(schema, gold)
            hardness = eval_hardness(g_sql)
            if hardness in ["hard", "extra"]:
                respond = text_to_sql(text, db_id)
                responds[id] = {"sql":respond, "db_id":db_id}
                golds[id] = {"sql":gold, "db_id":db_id}
                print(respond)
        except:
            print("error")
            continue
    #save the responds
    with open("./responds.json", "w") as f:
        json.dump(responds, f)
    with open("./golds.json", "w") as f:
        json.dump(golds, f)