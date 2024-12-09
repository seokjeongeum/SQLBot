import os
import json
import argparse


def read_data(data_type="train", project_dir="/home/hkkang/NL2QGM/"):
    data_dir = os.path.join(project_dir, "data/spider/")
    if data_type == 'train':
        data = json.load(open(os.path.join(data_dir, 'train_spider.json')))
        data += json.load(open(os.path.join(data_dir, 'train_others.json')))
    elif data_type == 'dev':
        data = json.load(open(os.path.join(data_dir, 'dev.json')))
    else:
        raise RuntimeError(f"Bad data_type:{data_type}")

    for idx, datum in enumerate(data):
        datum['idx'] = idx
        datum['db_id'] = datum['db_id'].lower()
        datum['question'] = datum['question'].lower()
        datum['query'] = datum['query'].lower()
    return data


def do_contain_key(keys, target):
    flags = [key in target for key in keys]
    return True in flags

def do_contain_all_keys(keys, target):
    flags = [key in target for key in keys]
    return False not in flags


def match(data, nl_keys, sql_keys, db_keys, nl_excludes, sql_excludes, db_excludes):
    tmps = []
    for datum in data:
        # Filter by db
        if (db_keys and datum['db_id'] not in db_keys):
            continue
        if (db_excludes and datum['db_id'] in db_excludes):
            continue

        # Filter by NL & SQL
        if (len(nl_keys) == 0 or do_contain_all_keys(nl_keys, datum['question'])) and (len(sql_keys)==0 or do_contain_all_keys(sql_keys, datum['query'])):
            if nl_excludes and do_contain_key(nl_excludes, datum['question']):
                continue
            if sql_excludes and do_contain_key(sql_excludes, datum['query']):
                continue
            tmps.append(datum)
    return tmps


def print_datum(datum):
    print(f"idx: {datum['idx']}")
    print(f"db_id: {datum['db_id']}")
    print(f"question: {datum['question']}")
    print(f"query: {datum['query']}\n")


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nl_key', type=str, nargs='+', default=[]) #NL key
    parser.add_argument('--sql_key', type=str, nargs='+', default=[]) # SQL key
    parser.add_argument('--db_key', type=str, nargs='+', default=[]) # DB key 
    parser.add_argument('--nl_exclude', type=str, nargs='+', default=[]) #NL exclude
    parser.add_argument('--sql_exclude', type=str, nargs='+', default=[]) #SQL exclude
    parser.add_argument('--db_exclude', type=str, nargs='+', default=[]) #DB exclude
    
    args = parser.parse_args()
    return args


def show_results(results):
    for result in results:
        print_datum(result)

    # print indices only
    print([result['idx'] for result in results])
    print(f"total:{len(results)}")


def search():
    args = read_args()
    data = read_data()
    results = match(data, args.nl_key, args.sql_key, args.db_key, args.nl_exclude, args.sql_exclude, args.db_exclude)
    show_results(results)

if __name__ == "__main__":
    search()
