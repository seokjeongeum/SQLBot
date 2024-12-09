import json

#file_path = "/root/NL2QGM/logdir/glove_run_no_join_cond/bs=20,lr=7.4e-04,end_lr=0e0,seed=2,join_cond=false/glove_run_no_Join_cond_true_1-step39200-eval.json"
file_path = "/root/NL2QGM/logdir/bert_run_no_join_cond/bs=1,lr=7.4e-04,bert_lr=1.0e-05,end_lr=0e0,seed=1,join_cond=false/ie_dirs/bert_run_true_1-step78000-eval.json"
table_path = "/root/NL2QGM/data/spider/tables.json"

def count_table(db):
    return len(db['table_names'])

def count_column(db):
    return len(db['column_names'])
    
def filter_by_schema_size(mode='table'):
    tables = json.load(open(table_path))
    dbs = {item['db_id']:item for item in tables}
    data = json.load(open(file_path))

    all = {}
    bad_cnt =0
    for idx, item in enumerate(data['per_item']):
        db_id = item['db_id']
        if not db_id:
            bad_cnt += 1
            continue
        func = count_table if mode == 'table' else count_column
        schema_idx = func(dbs[db_id])

        if schema_idx not in all:
            all[schema_idx] = {}
            all[schema_idx]['total'] = 1
            all[schema_idx]['correct'] = item['exact']
        else:
            all[schema_idx]['total'] += 1
            all[schema_idx]['correct'] += item['exact']


    sorted_keys = sorted(all)
    for key in sorted_keys:
        c = all[key]['correct']
        t = all[key]['total']
        print(f"size:{key} correct:{c} Total:{t} ({c/t*100}%)")


def filter_by_db_id():
    tables = json.load(open(table_path))
    dbs = {item['db_id']:item for item in tables}
    data = json.load(open(file_path))
    
    all = {}
    bad_cnt =0
    for idx, item in enumerate(data['per_item']):
        db_id = item['db_id']
        if not db_id:
            bad_cnt += 1
            continue
        schema_idx = count_table(dbs[db_id])

        if db_id not in all:
            all[db_id] = {}
            all[db_id]['total'] = 1
            all[db_id]['correct'] = item['exact']
            all[db_id]['table_len'] = count_table(dbs[db_id])
            all[db_id]['column_len'] = count_column(dbs[db_id])
        else:
            all[db_id]['total'] += 1
            all[db_id]['correct'] += item['exact']

    sorted_keys = sorted(all, key=(lambda k: all[k]['correct']/all[k]['total']), reverse=True)

    for key in sorted_keys:
        c = all[key]['correct']
        t = all[key]['total']
        tbl_len = all[key]['table_len']
        col_len = all[key]['column_len']
        print(f"size:{key}\t\ttable:{tbl_len}\t\tcolumn:{col_len} correct:{c} Total:{t} ({c/t*100}%)")

if __name__ == "__main__":
    filter_by_schema_size(mode='table')
    # filter_by_db_id()
