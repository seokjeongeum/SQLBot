import os
import sqlite3


dir = "/root/NL2QGM/create_db/"
file_template = "{}-{}.txt"
sql_template = "insert into {} values {};"

# dbs = {'wta_1': ['players', 'matches', 'rankings']}
# dbs = {'wta_1': ['matches', 'rankings']}
# dbs = {'car_1': ['continents', 'countries', 'car_makers', 'model_list', 'car_names', 'cars_data']}
dbs = {'world_1': ['city', 'sqlite_sequence', 'country', 'countrylanguage']}


def convert_number(item):
    need_change = True
    try:
        need_change = False
        item = int(item)
    except:
        pass
    if need_change:
        try:
            item = float(item)
        except:
            pass
    return item

if __name__ == "__main__":
    for db_name, table_names in dbs.items():
        print(f"Current DB: {db_name}")
        for table_name in table_names:
            print(f"Current Table:{table_name}")
            file_path = file_template.format(db_name, table_name)
            path = os.path.join(dir, file_path)
            sqls = []

            # Load and create
            print("Loading...")
            with open(path, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    items = line.split('|')
                    items = tuple(list(map(convert_number, items)))
                    sqls.append(sql_template.format(table_name, items))
            # Save
            print("Writing sqls...")
            new_path = path.replace('.txt', '.sql')
            with open(new_path, 'w') as f:
                for sql in sqls:
                    f.write(f"{sql}\n")
            print("done!")
        
            # Connect
            db_path = "/root/NL2QGM/data/spider_modified/database/bottom3_dbs/bottom3_dbs.sqlite"
            con = sqlite3.connect(db_path)
            # Insert
            cursor = con.cursor()
            for sql in sqls:
                # if 'insert into players' in sql:
                #     if int(sql.split('players values (')[1].split(',')[0]) <= 204047:
                #         print(f"Skipping: {sql}")
                #         continue
                print(f"Executing: {sql}")
                cursor.execute(sql)
            # Save and close
            con.commit()
            con.close()
