import os
import json

# ##20210420
# #BERT 
# file_path = "/mnt/nfs-server/backup/20210420/bert_run_true_1-step78000-eval-original.json"
# #GloVe
# file_path = "/mnt/nfs-server/backup/20210420/glove_run_no_join_cond_true_1-step38200-eval-original.json"

##20210414
#BERT
file_path = "/mnt/nfs-server/backup/20210414/bert_log/bert_run_true_1-step78000-eval.json"
#GloVe
file_path ="/mnt/nfs-server/backup/20210414/glove_log/glove_run_no_Join_cond_true_1-step39200-eval.json"
new_file_path ="/mnt/nfs-server/backup/20210414/glove_log/ie_dirs/glove_run_no_Join_cond_true_1-step39200-eval.json"

target_dbs = ['poker_player', 'singer', 'orchestra', 'concert_singer', 'network_1', 'wta_1', 'car_1', 'world_1', 'real_estate_properties']

if __name__ == "__main__":
    data = json.load(open(file_path))
    new_data = json.load(open(new_file_path))

    print(f"Total acc:{data['total_scores']['all']['exact']} -> {new_data['total_scores']['all']['exact']}")
    data = list(filter(lambda k: k['db_id'] in target_dbs, data['per_item']))
    new_data = new_data['per_item']

    assert len(data) == len(new_data), f"{len(data)}!={len(new_data)}"

    db_dic = {}
    new_db_dic = {}
    for item, new_item in zip(data, new_data):
        db_id = item['db_id']
        assert item['gold'] == new_item['gold']
        if db_id in db_dic:
            db_dic[db_id]['correct'] += item['exact']
            db_dic[db_id]['total'] += 1

            new_db_dic[db_id]['correct'] += new_item['exact']
            new_db_dic[db_id]['total'] += 1

        else:
            db_dic[db_id] = {'total': 1 ,'correct': item['exact']}
            new_db_dic[db_id] = {'total': 1 ,'correct': new_item['exact']}
        
        
    for db_id in target_dbs:
        item = db_dic[db_id]
        acc = item['correct'] / item['total']
        print(f"{db_id}:")
        print(f"\tacc:{acc} ({item['correct']}/{item['total']})\n")

    print('\n\nNew:')
    for db_id in target_dbs:
        item = new_db_dic[db_id]
        acc = item['correct'] / item['total']
        print(f"{db_id}:")
        print(f"\tacc:{acc} ({item['correct']}/{item['total']})\n")
