import json
import unittest

from ratsql.datasets.spider_lib.evaluation import Evaluator, build_foreign_key_map_from_json
from ratsql.datasets.spider_lib import evaluation_original

# XXX: check unittest coding convensions
class Test_Evaluation(unittest.TestCase):
    def _prepare_evaluator(self, 
                            grammar='postgresql', 
                            db_dir = 'data/debug/database',
                            tables_path = 'data/debug/tables.json',
                            db_type = 'sqlite',
                            ):
        kmaps = build_foreign_key_map_from_json(tables_path)
        etype = 'match'

        with open(tables_path) as f:
            tables = json.load(f)

        evaluator = Evaluator(db_dir, kmaps, tables, etype, db_type=db_type, grammar=grammar)

        return evaluator

    def _prepare_original_evaluator(self, 
                                    grammar='postgresql', 
                                    db_dir = 'data/debug/database',
                                    tables_path = 'data/debug/tables.json',
                                    db_type = 'sqlite',
                                    ):
        db_dir = 'data/debug/database'
        tables_path = 'data/debug/tables.json'
        kmaps = evaluation_original.build_foreign_key_map_from_json(tables_path)
        etype = 'match'
        db_type = 'sqlite'

        with open(tables_path) as f:
            tables = json.load(f)

        evaluator = evaluation_original.Evaluator(db_dir, kmaps, tables, etype, db_type=db_type)

        return evaluator

    def test_evaluate_one_1(self):
        evaluator = self._prepare_evaluator(grammar='postgresql',
                                            db_dir="user=postgres password=postgres host=localhost port=5435",
                                            tables_path='data/samsung/tables.json',
                                            db_type='postgresql')
        
        db_name = 'samsung'
        gold = "select distinct root_lot_id from QUALITY.m_fab_tracking where PROCESS_ID = 'PPIT'"
        predicted = "SELECT * FROM quality.m_fab_tracking WHERE quality.m_fab_tracking.root_lot_id = 'terminal'"
        result = evaluator.evaluate_one(db_name, gold, predicted)
        self.assertEqual(result['exact'], 0)

    def test_evaluate_one_2_postgres(self):
        evaluator = self._prepare_evaluator(grammar='postgresql')
        
        db_name = 'employee_hire_evaluation'
        gold = "select * from employee"
        predicted = "select * from hiring"
        result = evaluator.evaluate_one(db_name, gold, predicted)
        self.assertEqual(result['exact'], 0)

    def test_evaluate_one_2_spider(self):
        evaluator = self._prepare_evaluator(grammar='spider')
        
        db_name = 'employee_hire_evaluation'
        gold = "select * from employee"
        predicted = "select * from hiring"
        result = evaluator.evaluate_one(db_name, gold, predicted)
        self.assertEqual(result['exact'], 0)        

    def test_evaluate_one_3(self):
        evaluator = self._prepare_original_evaluator(grammar='spider')
        
        db_name = 'employee_hire_evaluation'
        gold = "select * from employee"
        predicted = "select * from hiring"
        result = evaluator.evaluate_one(db_name, gold, predicted)
        self.assertEqual(result['exact'], 0)

if __name__ == '__main__':
    unittest.main()