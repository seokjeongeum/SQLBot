import json

# Heuristic
file_1_name = "/root/NL2QGM/logdir/glove_run_yes_join_cond/bs=20,lr=7.4e-04,end_lr=0e0,seed=2,join_cond=true/glove_run_yes_join_cond_true_1-step17200-eval-heuristic.json"

# NN-based
file_2_name = "/root/NL2QGM/logdir/glove_run_yes_join_cond/bs=20,lr=7.4e-04,end_lr=0e0,seed=2,join_cond=true/glove_run_yes_join_cond_true_1-step17200-eval-nn.json"

dic_1 = json.load(open(file_1_name))
dic_2 = json.load(open(file_2_name))

diff_table_cnt = 0
h_right_table = 0
nn_right_table = 0

diff_join_cnt = 0
h_right_join = 0 
nn_right_join = 0


assert len(dic_1['per_item']) == len(dic_2['per_item'])
for idx, (item1, item2) in enumerate(zip(dic_1['per_item'], dic_2['per_item'])):
    assert item1['gold'] == item2['gold']
    if item1['exact'] == 1 or item2['exact'] == 1:
        if item1['partial']['table']['acc'] != item2['partial']['table']['acc']:
            print(f"\nIdx:{idx}")
            print("Different table result!")
            print(f"Gold:      {item1['gold']}")
            print(f"Heuristic: {item1['predicted']}")
            print(f"NN:        {item2['predicted']}")
            diff_table_cnt += 1
            if item1['partial']['table']['acc']:
                print("Heuristic is right")
                h_right_table += 1
            else:
                print("NN is right")
                nn_right_table += 1
        
        if item1['partial']['join_condition']['acc'] != item2['partial']['join_condition']['acc']:
            if item1['partial']['table']['acc'] == item2['partial']['table']['acc']:
                print(f"\nIdx:{idx}")
            else:
                print("")
            print("Different join condition result!")
            print(f"Gold:      {item1['gold']}")
            print(f"Heuristic: {item1['predicted']}")
            print(f"NN:        {item2['predicted']}")
            diff_join_cnt += 1
            if item1['partial']['join_condition']['acc']:
                print("Heuristic is right")
                h_right_join += 1
            else:
                print("NN is right")
                nn_right_join += 1

print(f"\nStatistic: (Total: {len(dic_1['per_item'])})")
print(f"diff_table_cnt:{diff_table_cnt} heuristic_right:{h_right_table} NN_right:{nn_right_table}")
print(f"diff_join_cnt: {diff_join_cnt} heuristic_right:{h_right_join} NN_right:{nn_right_join}")

print("\nDone!")
