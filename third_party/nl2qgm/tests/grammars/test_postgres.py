import json
import unittest
from ratsql.datasets.spider import load_tables
from ratsql.datasets.spider_lib.process_sql_postgres import Schema, get_schema, get_sql
from ratsql.grammars.postgres import PostgresLanguage, PostgresUnparser

# XXX: check unittest coding convensions
class Test_PostgresLanguage(unittest.TestCase):
    def _test_query(self, query):
        tables_file = '/mnt/sdc/jjkim/NL2QGM/data/samsung/tables.json'
        with open(tables_file, 'r') as f:
            tables = json.load(f)
        db = 'dbname=samsung user=postgres password=postgres host=localhost port=5435'
        schema = Schema(get_schema(db, db_type='postgresql'), tables=tables[0])
        sql = get_sql(schema, query)
        print(sql)

        return sql

    def _test_sql(self, sql):
        parsed_sql = PostgresLanguage().parse(sql, 'train')
        print(parsed_sql)
        return parsed_sql

    def test_unnest_1(self):
        query = """
            select 
                eqp_id, 
                unnest(item_id) as itemID 
            from 
                quality.m_fab_dcop_met 
            where 
                line_id = 'KFBG' and 
                root_lot_id = 'GKG706' 
            order by 
                eqp_id
        """
        sql = self._test_query(query)
        self._test_sql(sql)
        

    def test_unnest_2(self):
        query = """
            select 
                lot_id, 
                wafer_id, 
                unnest(chip_x_pos) chip_x_pos, 
                unnest(chip_y_pos) chip_y_pos, 
                unnest(good_bin) good_bin
            from 
                quality.m_eds_chip_bin
            where 
                process_id = 'KCAK'
            order by 
                lot_id, 
                wafer_id, 
                chip_x_pos, 
                chip_y_pos
        """
        sql = self._test_query(query)
        self._test_sql(sql)

    def test_concat_1(self):
        query = """
            select 
                count (distinct (step_seq||','||item_id))
            from 
                master.m_defect_item
            where 
                line_id = 'KFBH' and 
                process_id = 'KGKV'
        """
        sql = self._test_query(query)
        self._test_sql(sql)

    def test_concat_2(self):
        query = """
            select 
                *
            from 
                quality.m_fab_tracking
            where 
                lot_id||'_'||part_id in (
                    select 
                        distinct lot_id||'_'||part_id
                    from 
                        quality.m_defect_chip
                    where 
                        part_id = 'KHAA84901B-GEL'
                    group by 
                        lot_id||'_'||part_id
                    having 
                        avg(total_dft_cnt) > 100
                )
        """
        sql = self._test_query(query)
        self._test_sql(sql)

    def test_array_len_1(self):
        query = """
            select 
                lot_id, wafer_id, array_length(BIN_NO,1)
            from 
                quality.m_eds_chip_bin
            where 
                process_id = 'KCAK'
            order by 
                lot_id, wafer_id
        """
        sql = self._test_query(query)
        self._test_sql(sql)

    def test_current_timestamp_1(self):
        query = """
            select *
            from master.m_defect_item
            where line_id = 'KFBG'
            and process_id = 'KGKV'
            and last_update_time > current_timestamp + '-7 days'
        """
        sql = self._test_query(query)
        return self._test_sql(sql)

class Test_PostgresUnparser(unittest.TestCase):
    def _test_query(self, query):
        tables_file = '/mnt/sdc/jjkim/NL2QGM/data/samsung/tables.json'
        with open(tables_file, 'r') as f:
            tables = json.load(f)
        db = 'dbname=samsung user=postgres password=postgres host=localhost port=5435'
        schema = Schema(get_schema(db, db_type='postgresql'), tables=tables[0])
        sql = get_sql(schema, query)
        return sql

    def _test_sql(self, sql):
        parsed_sql = PostgresLanguage().parse(sql, 'train')
        return parsed_sql

    def _test_unparse(self, tree):
        tables_file = '/mnt/sdc/jjkim/NL2QGM/data/samsung/tables.json'
        schemas, _ = load_tables([tables_file], False)
        schema = schemas['samsung']

        lang = PostgresLanguage()
        unparser = PostgresUnparser(lang.ast_wrapper, schema, lang.factorize_sketch)
        return unparser.unparse_sql(tree)

    def test_unnest_1(self):
        query = """
            select 
                eqp_id, 
                unnest(item_id) as itemID 
            from 
                quality.m_fab_dcop_met 
            where 
                line_id = 'KFBG' and 
                root_lot_id = 'GKG706' 
            order by 
                eqp_id
        """
        print(query)
        sql = self._test_query(query)
        tree = self._test_sql(sql)
        print(self._test_unparse(tree))

    def test_unnest_2(self):
        query = """
            select 
                lot_id, 
                wafer_id, 
                unnest(chip_x_pos) chip_x_pos, 
                unnest(chip_y_pos) chip_y_pos, 
                unnest(good_bin) good_bin
            from 
                quality.m_eds_chip_bin
            where 
                process_id = 'KCAK'
            order by 
                lot_id, 
                wafer_id, 
                chip_x_pos, 
                chip_y_pos
        """
        print(query)
        sql = self._test_query(query)
        tree = self._test_sql(sql)
        print(self._test_unparse(tree))

    def test_concat_1(self):
        query = """
            select 
                count (distinct (step_seq||','||item_id))
            from 
                master.m_defect_item
            where 
                line_id = 'KFBH' and 
                process_id = 'KGKV'
        """
        print(query)
        sql = self._test_query(query)
        tree = self._test_sql(sql)
        print(self._test_unparse(tree))

    def test_concat_2(self):
        query = """
            select *
            from 
                quality.m_fab_tracking
            where 
                lot_id||'_'||part_id in (
                    select 
                        distinct lot_id||'_'||part_id
                    from 
                        quality.m_defect_chip
                    where 
                        part_id = 'KHAA84901B-GEL'
                    group by 
                        lot_id||'_'||part_id
                    having 
                        avg(total_dft_cnt) > 100
                )
        """
        print(query)
        sql = self._test_query(query)
        tree = self._test_sql(sql)
        print(self._test_unparse(tree))

    def test_array_len_1(self):
        query = """
            select 
                lot_id, wafer_id, array_length(BIN_NO,1)
            from 
                quality.m_eds_chip_bin
            where 
                process_id = 'KCAK'
            order by 
                lot_id, wafer_id
        """
        print(query)
        sql = self._test_query(query)
        tree = self._test_sql(sql)
        print(self._test_unparse(tree))

    def test_current_timestamp_1(self):
        query = """
            select *
            from master.m_defect_item
            where line_id = 'KFBG'
            and process_id = 'KGKV'
            and last_update_time > current_timestamp + '-7 days'
        """
        print(query)
        sql = self._test_query(query)
        tree = self._test_sql(sql)
        print(self._test_unparse(tree))

    def test_is_not_null_1(self):
        query = """
            select *
            from 
                master.m_defect_item
            where 
                line_id = 'KFBH' and 
                process_id = 'KCHY' and 
                target is not null
        """
        print(query)
        sql = self._test_query(query)
        tree = self._test_sql(sql)
        print(self._test_unparse(tree))

    def test_is_not_null_2(self):
        query = """
            select 
                line_id, step_seq, item_id, spec_high 
            from 
                master.m_defect_item 
            where 
                process_id = 'KGKV' and 
                spec_high is not null
        """
        print(query)
        sql = self._test_query(query)
        tree = self._test_sql(sql)
        print(self._test_unparse(tree))
        

if __name__ == '__main__':
    unittest.main()