import os 
import json

target_dbs = [['poker_player', 'singer', 'orchestra'], ['concert_singer', 'network_1', 'wta_1'], ['car_1', 'world_1', 'real_estate_properties']]
db_mapping = {'top3_dbs': ['poker_player', 'singer', 'orchestra'], 'middle3_dbs': ['concert_singer', 'network_1', 'wta_1'], 'bottom3_dbs': ['car_1', 'world_1', 'real_estate_properties']}

# Glove: left is with bigger scheam, right is with smaller schema
# result_1 = "/root/NL2QGM/logdir/glove_run_no_join_cond/bs=20,lr=7.4e-04,end_lr=0e0,seed=2,join_cond=false/ie_dirs/glove_run_no_Join_cond_true_1-step40000-eval.json"
# result_2 = "/root/NL2QGM/logdir/glove_run_no_join_cond/bs=20,lr=7.4e-04,end_lr=0e0,seed=2,join_cond=false/glove_run_no_Join_cond_true_1-step39200-eval.json"
# result_1 = "/mnt/nfs-server/backup/20210414/glove_log/ie_dirs/glove_run_no_Join_cond_true_1-step39200-eval.json"
# result_2 = "/mnt/nfs-server/backup/20210414/glove_log/glove_run_no_Join_cond_true_1-step39200-eval.json"

# BERT: left is with bigger schema, right is with smaller schema
# result_1 = "/root/NL2QGM/logdir/bert_run_no_join_cond/bs=1,lr=7.4e-04,bert_lr=1.0e-05,end_lr=0e0,seed=1,join_cond=false/ie_dirs/bert_run_true_1-step78000-eval.json"
# result_2 = "/root/NL2QGM/logdir/bert_run_no_join_cond/bs=1,lr=7.4e-04,bert_lr=1.0e-05,end_lr=0e0,seed=1,join_cond=false/bert_run_true_1-step78000-eval.json"

# result_1 = "/mnt/nfs-server/backup/20210414/bert_log/ie_dirs/bert_run_true_1-step78000-eval-filtered_examples.json"
# result_2 = "/mnt/nfs-server/backup/20210414/bert_log/bert_run_true_1-step78000-eval.json"


data_1 = json.load(open(result_1))['per_item']
data_2 = json.load(open(result_2))['per_item']

def find_example(target, data):
    for item in data:
        if item['gold'] == target['gold'] and item['db_id'] in db_mapping[target['db_id']]:
            data.remove(item)
            return item
    raise RuntimeError("Should not be here")

def left_percentage(item):
    return (item['left_correct'] + item['both_correct']) / item['cnt']

def right_percentage(item):
    return (item['right_correct'] + item['both_correct']) / item['cnt']


if __name__ == "__main__":
    # Alighn data
    tmp = []
    for item in data_1:
        found_item = find_example(item,  data_2)
        tmp.append(found_item)
    data_2 = tmp

    # compare_data
    left = []
    right = []
    both = []
    left_correct = 0
    right_correct = 0
    both_correct =  0
    db_stat = {}
    assert len(data_1) == len(data_2)
    for idx, (datum_1, datum_2) in enumerate(zip(data_1, data_2)):
        # For db_id
        db_id = datum_2['db_id']
        if db_id not in db_stat:
            db_stat[db_id] = {"cnt": 0, "left_correct": 0, "right_correct": 0, "both_correct": 0}

        db_stat[db_id]['cnt'] +=1
        if datum_1['exact'] and datum_2['exact']:
            both_correct += 1
            both += [(datum_1, datum_2)]
            db_stat[db_id]['both_correct'] += 1
        elif datum_1['exact']:
            left_correct += 1
            left += [(datum_1, datum_2)]

            db_stat[db_id]['left_correct'] += 1
        elif datum_2['exact']:
            right_correct += 1
            right += [(datum_1, datum_2)]

            db_stat[db_id]['right_correct'] += 1
        else:
            pass # Both wrong


    print(f"\nTotal cnt:{len(data_1)} Both_correct:{both_correct} left_correct:{left_correct} right_correct:{right_correct}")

    print("")

    for key ,item in db_stat.items():
        print(f"DB:{key} cnt:{item['cnt']} both:{item['both_correct']} left:{item['left_correct']} \
        ({left_percentage(item)}) right:{item['right_correct']} ({right_percentage(item)})")

    for dbs in target_dbs:
        total_cnt = 0
        left_cnt = 0
        right_cnt = 0
        for db in dbs:
            total_cnt += db_stat[db]['cnt']
            left_cnt += db_stat[db]['both_correct'] + db_stat[db]['left_correct']
            right_cnt += db_stat[db]['both_correct'] + db_stat[db]['right_correct']
        print(f"\nDB:{dbs}")
        print(f"Total:{total_cnt} left:{left_cnt} ({left_cnt/total_cnt}) right:{right_cnt} ({right_cnt/total_cnt})")
