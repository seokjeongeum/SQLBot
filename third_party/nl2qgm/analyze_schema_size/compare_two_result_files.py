import os
import json

# BERT
# orig_result_path = "/mnt/nfs-server/backup/20210414/bert_log/ie_dirs/bert_run_true_1-step78000-eval-filtered_examples.json"
# orig_result_path = "/mnt/nfs-server/backup/20210420/bert_run_true_1-step78000-eval-original.json"
# extended_result_path = "/mnt/nfs-server/backup/20210420/bert_run_true_1-step78000-eval-extended.json"

# GloVe
orig_result_path = "/mnt/nfs-server/backup/20210420/glove_run_no_join_cond_true_1-step39200-eval-original.json"
extended_result_path = "/mnt/nfs-server/backup/20210420/glove_run_no_join_cond_true_1-step39200-eval-extended.json"

orig_table_path = "/root/NL2QGM/data/spider/tables.json"
extended_table_path = "/root/NL2QGM/data/spider/tables_extended.json"

group_info = {  "top3_dbs":["poker_player", "singer", "orchestra"],
                "middle3_dbs": ["concert_singer", "network_1", "wta_1"],
                "bottom3_dbs": ["car_1", "world_1", "real_estate_properties"] }


def load_result_file(eval_path):
    infer_path = eval_path.replace("-eval-", "-infer-").replace(".json", ".jsonl")
    eval_result = json.load(open(eval_path))
    assert os.path.isfile(infer_path), f"{infer_path} does not exist!"

    # Load infer result
    with open(infer_path) as f:
        infer_result = [json.loads(line) for line in f.readlines()]
    assert len(infer_result) == len(eval_result['per_item']), f"Length different! {len(infer_result)} : {len(eval_result['per_item'])}"
    
    # Append NL query
    for idx in range(len(infer_result)):
        question = infer_result[idx]['beams'][0]['orig_question']
        eval_result['per_item'][idx]['question'] = question

    return eval_result


def get_db_mapping(items):
    dbs = list(set([item['db_id'] for item in items]))
    
    # Create mapping
    db_mapping = {}
    for db_name in dbs:
        assert '__' in db_name, f"db_name:{db_name}"
        base_db_name = db_name.split('__')[0]
        if base_db_name not in db_mapping:
            db_mapping[base_db_name] = [db_name]
        else:
            db_mapping[base_db_name] += [db_name]
    
    # Sorting values
    for key in db_mapping.keys():
        db_mapping[key] = sorted(db_mapping[key])

    return db_mapping


def group_queries(items):
    mapping = {}
    for item in items:
        db_id = item['db_id']
        if db_id in mapping:
            mapping[db_id] += [item]
        else:
            mapping[db_id] = [item]
    return mapping


def calculate_acc(items):
    return sum([item['exact'] for item in items]) / len(items)


def print_acc_col_tab_query_num(table_infos, db_id, items, end='\n'):
    total_acc = calculate_acc(items)
    col_num = len(table_infos[db_id]['column_names'])
    tab_num = len(table_infos[db_id]['table_names'])
    total_query = len(items)
    correct_query = len(list(filter(lambda k: k['exact'] == 1, items)))
    print("{}: Acc:{:.2f} ({}/{}) Tab:{} Col:{}".format(db_id, total_acc, correct_query, total_query, tab_num, col_num), end=end)


def compare_per_group_acc(table_infos, db_mapping, orig_data, extended_data):
    print("Per Group Acc:")
    for group_name, group_dbs in group_info.items():
        # Get all queries for this group
        target_orig_queries = []
        for db_name in group_dbs:
            target_orig_queries += orig_data[db_name]
        
        # Calculate Total accuarcy
        acc = calculate_acc(target_orig_queries)
        correct_queries_num = len(list(filter(lambda k: k['exact'] == 1, target_orig_queries)))
        print("Group:{} Acc:{} ({}/{})".format(group_name, acc, correct_queries_num, len(target_orig_queries)))

        # Per extended DB Group
        max_len = max([len(db_mapping[db_name]) for db_name in group_dbs])
        for idx in range(1,max_len+1):
            # Get all queries for this extended group
            target_extended_queries = []
            for db_name in group_dbs:
                extended_db_name = f"{db_name}__{idx}"
                if extended_db_name in extended_data.keys():
                    target_extended_queries += extended_data[extended_db_name]
                    
            # Calculate Total accuarcy
            acc = calculate_acc(target_extended_queries)
            correct_queries_num = len(list(filter(lambda k: k['exact'] == 1, target_extended_queries)))
            print("\tGroup_idx:{} Acc:{} ({}/{})".format(idx, acc, correct_queries_num, len(target_extended_queries)))
        print('')            
    print('\n')


def compare_per_db_accuracy(table_infos, db_mapping, orig_data, extended_data):
    print("Per DB Acc:")
    for base_db, base_queries in orig_data.items():
        print_acc_col_tab_query_num(table_infos, base_db, base_queries)
        
        # Info of Corresponding extended DB
        for extended_db in db_mapping[base_db]:
            print('\t', end='')
            print_acc_col_tab_query_num(table_infos, extended_db, extended_data[extended_db])
        print('')


def find_item_by_gold_sql(items, question):
    items = list(filter(lambda k: k['question'] == question, items))
    assert len(items) == 1, "Same SQL item exists..."
    return items[0]


def compare_per_added_db_accuracy(table_infos, db_mapping, orig_data, extended_data):
    print("Per Added DB Acc:")
    new = {0: {'total': 0, 'correct': 0}}
    for base_db, base_queries in orig_data.items():
        for base_query in base_queries:
            new[0]['correct'] += base_query['exact']
            new[0]['total'] += 1
    
    for db_id, base_queries in extended_data.items():
        added_num = db_id.split('__')[1]
        if added_num not in new:
            new[added_num] = {'total':0, 'correct':0}
        for base_query in base_queries:
            new[added_num]['correct'] += base_query['exact']
            new[added_num]['total'] += 1

    for key in new.keys():
        acc = new[key]['correct'] / new[key]['total']
        print(f"\tadded_num:{key} acc:{acc} ({new[key]['correct']} / {new[key]['total']})")


def compare_per_query_accuracy(table_infos, db_mapping, orig_data, extended_data):
    print("Per Query Acc:")
    for base_db, base_queries in orig_data.items():
        print_acc_col_tab_query_num(table_infos, base_db, base_queries)

        # Per query acc
        for query_idx, query in enumerate(base_queries):
            print(f"\tIdx:{query_idx} Gold:{query['gold']}")
            for db_idx, extended_db in enumerate(db_mapping[base_db]):
                target_item = find_item_by_gold_sql(extended_data[extended_db], query['question'])
                print(f"\t\tDB:{db_idx+1} Correct:{target_item['exact']} Pred:{target_item['predicted']}")
            print('\n')
        print('')


if __name__ == "__main__":
    # Load table info
    tables = json.load(open(orig_table_path)) + json.load(open(extended_table_path))
    table_infos = {item['db_id']: item for item in tables}

    # Load Results
    orig_results = load_result_file(orig_result_path)
    extended_results = load_result_file(extended_result_path)
    print(f"original results: {len(orig_results['per_item'])} extended results: {len(extended_results['per_item'])}")

    # Create DB mapping
    db_mapping = get_db_mapping(extended_results['per_item'])
    for base_db, extended_dbs in db_mapping.items():
        print(f"base_db:{base_db} extended_db_num:{len(extended_dbs)}")

    # group queries by db_id
    orig_queries = group_queries(orig_results['per_item'])
    extended_queries = group_queries(extended_results['per_item'])

    # filter target dbs
    orig_queries = dict(filter(lambda k: k[0] in db_mapping.keys(), orig_queries.items()))

    # Compare total accuracy per Group
    compare_per_group_acc(table_infos, db_mapping, orig_queries, extended_queries)

    # Compare total accuracy per DB
    compare_per_db_accuracy(table_infos, db_mapping, orig_queries, extended_queries)

    # Compare per query accuracy
    compare_per_query_accuracy(table_infos, db_mapping, orig_queries, extended_queries)

    # All average accuracy per added db
    compare_per_added_db_accuracy(table_infos, db_mapping, orig_queries, extended_queries)
