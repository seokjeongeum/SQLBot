import json

dev_file = "/root/NL2QGM/data/spider/dev.json"

dev = json.load(open(dev_file))

cnt=0
for idx, item in enumerate(dev):
    if item['db_id'] == "flight_2":
        if ' JOIN ' in item['query']:
            print(f"\nidx:{idx} {item['query']}")
            cnt += 1

print(f"Total number:{cnt}")
