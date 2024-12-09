import json

train_file_paths = ["/root/NL2QGM/data/spider/train_spider.json", "/root/NL2QGM/data/spider/train_others.json"]
dev_file_paths = ["/root/NL2QGM/data/spider/dev.json"]

def has_nested_query(query: str):
    return '( SELECT' in query or '(SELECT' in query 

def has_set_operator(query: str):
    return ' INTERSECT ' in query or ' UNION ' in query or ' EXCEPT ' in query


def count_subquery(file_paths):
    subquery_cnt = 0 
    total_cnt = 0
    for file_path in file_paths:
        # for file_path in dev_file_paths:
        data = json.load(open(file_path))
        for idx, item in enumerate(data):
            total_cnt += 1
            query = item['query']
            if has_nested_query(query):
                subquery_cnt += 1
            else:
                if not has_set_operator(query) and query.count("SELECT") > 1:
                    print(f"Something wrong: {query}")
        
    print(f"Total nested query:{subquery_cnt} ({subquery_cnt/total_cnt*100}%)")


if __name__ == "__main__":
    print("\nTrain:")
    count_subquery(train_file_paths)
    print("\nDev:")
    count_subquery(dev_file_paths)
