import os
import json


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)

def read_in_result(result_path):
    result = read_json(result_path)['per_item']
    # Append idx
    for idx, item in enumerate(result):
        if 'idx' not in item:
            item['idx'] = idx
    return result

def get_correct_data(data):
    return list(filter(lambda x: x['exact'], data))

def get_incorrect_data(data):
    return list(filter(lambda x: not x['exact'], data))

def print_datum(datum):
    print(f"Idx: {datum['idx']}")
    print(f"NL: {datum['orig_question']}")
    print(f"GOLD: {datum['gold']}")
    print(f"PRED: {datum['predicted']}")
    print('')

if __name__ == "__main__":
    result_path = "/data/hkkang/NL2QGM/logdir/samsung-addop-hkkang/bs=1,lr=7.4e-04,bert_lr=3.0e-06,end_lr=0e0,seed=0,join_cond=false/ie_dirs/samsung-addop-hkkang-step_54750-eval.json"
    result_path = "/data/hkkang/NL2QGM/logdir/samsung-addop-hkkang-testing/bs=1,lr=7.4e-04,bert_lr=3.0e-06,end_lr=0e0,seed=0,join_cond=false/ie_dirs/samsung-addop-hkkang-testing-step_54750-eval.json"

    result_data = read_in_result(result_path)
    wrong_data = get_incorrect_data(result_data)
    right_data = get_correct_data(result_data)

    assert len(wrong_data) + len(right_data) == len(result_data), "Something is wrong with counting"

    print(f"Correct Data: (len:{len(right_data)})")
    for datum in right_data:
        print_datum(datum)
    
    print(f"\nIncorrect Data: (len:{len(wrong_data)})")
    for datum in wrong_data:
        print_datum(datum)
    