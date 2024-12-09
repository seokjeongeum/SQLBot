import json
import unittest

from ratsql.datasets.spider_lib.process_sql import Schema, get_schema, get_sql


# XXX: check unittest coding convensions
class Test_GetSchema(unittest.TestCase):
    def test_get_schema(self):
        db = 'dbname=samsung user=postgres password=postgres host=localhost port=5435'
        schema = get_schema(db, db_type='postgresql')
        print(schema)
    
class Test_GetSql(unittest.TestCase):
    def _test_query(self, query):
        tables_file = '/mnt/sdc/jjkim/NL2QGM/data/samsung/tables.json'
        with open(tables_file, 'r') as f:
            tables = json.load(f)
        db = 'dbname=samsung user=postgres password=postgres host=localhost port=5435'
        schema = Schema(get_schema(db, db_type='postgresql'), tables=tables[0])
        sql = get_sql(schema, query)

        return sql

    def test_etc(self):
        query = """SELECT * FROM master.m_fab_item_met WHERE master.m_fab_item_met.line_id = 'terminal' LIMIT 1"""
        sql = self._test_query(query)
        print(sql)


if __name__ == '__main__':
    unittest.main()
