import json

file_path = "/root/NL2QGM/logdir/glove_run_no_join_cond/bs=20,lr=7.4e-04,end_lr=0e0,seed=1,join_cond=false/eval_result_39200.json"

def has_nested_query(query: str) -> bool :
    return '( SELECT' in query or '(SELECT' in query 

def has_set_operator(query: str) -> bool:
    return ' INTERSECT ' in query or ' UNION ' in query or ' EXCEPT ' in query


if __name__ == "__main__":
    data = json.load(open(file_path))
    
    correct_cnt = 0
    total_cnt = 0
    for idx, item in enumerate(data['per_item']):
        query = item['gold']
        if has_nested_query(query):
            total_cnt += 1
            if item['exact']:
                correct_cnt +=1 
        else:
            if not has_set_operator(query) and query.count("SELECT") > 1:
                print(f"Something wrong: {query}")
    
    print(f"Total nested query:{total_cnt}")
    print(f"Correct nested query:{correct_cnt}({correct_cnt/total_cnt*100}%)")
