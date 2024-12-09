import json

file_path = "/root/NL2QGM/logdir/glove_run_no_join_cond/bs=20,lr=7.4e-04,end_lr=0e0,seed=1,join_cond=false/eval_result_39200.json"

data = json.load(open(file_path))

all_cnt = 0
all_correct_cnt = 0
special_cnt = 0
special_correct_cnt = 0
for idx, item in enumerate(data['per_item']):
    query = item['gold']
    # For all join query
    if " JOIN " in query:
        all_cnt += 1
        if item['exact']:
            all_correct_cnt += 1
        # If IUE or Nested
        if " INTERSECT " in query or " UNION " in query or " EXCEPT " in query or "(SELECT " in query:
            pass
        else:
            if item['exact']:
                special_correct_cnt += 1
            special_cnt += 1

print(f"all_cnt:{all_correct_cnt}/{all_cnt}")
print(f"special_cnt:{special_correct_cnt}/{special_cnt}")
