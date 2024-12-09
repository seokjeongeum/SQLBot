import json

# Heuristic
file_1_name = "/root/NL2QGM/logdir/glove_run_yes_join_cond/bs=20,lr=7.4e-04,end_lr=0e0,seed=2,join_cond=true/glove_run_yes_join_cond_true_1-step17200-eval-heuristic.json"

# NN-based
file_2_name = "/root/NL2QGM/logdir/glove_run_yes_join_cond/bs=20,lr=7.4e-04,end_lr=0e0,seed=2,join_cond=true/glove_run_yes_join_cond_true_1-step17200-eval-nn.json"

file_path = file_2_name

data = json.load(open(file_path))['per_item']


cnt = 0
cnt_total = 0
for datum in data:
    if ' JOIN ' in datum['gold']:
        cnt_total += 1
        if datum['partial']['join_condition']['acc']:
            cnt += 1

print(f"cnt:{cnt} cnt_total:{cnt_total} percentage:{cnt/cnt_total}")

