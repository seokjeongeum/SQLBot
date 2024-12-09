import os
import json

ori_result_path = "/mnt/nfs-server/backup/20210414/bert_log/bert_run_true_1-step78000-eval.json"
new_result_path = "/mnt/nfs-server/backup/20210414/bert_log/ie_dirs/bert_run_true_1-step78000-eval-fixed-tables.json"

target_dbs = ['poker_player', 'singer', 'orchestra', 'concert_singer', 'network_1', 'wta_1', 'car_1', 'world_1', 'real_estate_properties']


def main():
    pass


if __name__ == "__main__":
    ori_data = json.load(open(ori_result_path))['per_item']
    new_data = json.load(open(new_result_path))['per_item']

    ori_data = list(filter(lambda k: k['db_id'] in target_dbs, ori_data))
    for idx, (ori, new) in enumerate(zip(ori_data, new_data)):
        assert ori['gold'] == new['gold']
        if ori['exact'] != new['exact'] and ori['db_id'] in ["singer"]:
            print(f"Idx:{idx} {ori['db_id']}")
            print(f"Gold:{ori['gold']}")
            print(f"ori:{ori['predicted']}")
            print(f"new:{new['predicted']}\n")
