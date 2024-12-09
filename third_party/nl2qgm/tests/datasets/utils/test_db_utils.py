import unittest
from ratsql.datasets.utils.db_utils import connect, fetch_col_names, fetch_table_names


class Test_Connect(unittest.TestCase):
    def test_connect(self):
        db = 'dbname=samsung user=postgres password=postgres host=localhost port=5435'
        with connect(db, db_type='postgresql') as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT 1')
    
class Test_FetchTableNames(unittest.TestCase):
    def test_fetch_table_names_postgres(self):
        db = 'dbname=samsung user=postgres password=postgres host=localhost port=5435'
        with connect(db, db_type='postgresql') as conn:
            cursor = conn.cursor()
            tables = fetch_table_names(cursor, db_type='postgresql')
            print(tables)
            self.assertEqual(len(tables), 10)

class Test_FetchColNames(unittest.TestCase):
    def test_fetch_col_names_postgres(self):
        db = 'dbname=samsung user=postgres password=postgres host=localhost port=5435'
        with connect(db, db_type='postgresql') as conn:
            cursor = conn.cursor()
            cols = fetch_col_names(cursor, 'master.m_defect_item', db_type='postgresql')
            self.assertEqual(len(cols), 11)


if __name__ == '__main__':
    unittest.main()