import json
import unittest
from ratsql.datasets.spider_lib.process_sql_postgres import get_schema, get_sql, Schema


# XXX: check unittest coding convensions
class Test_GetSql(unittest.TestCase):
    def _test_query(self, query):
        tables_file = '/mnt/sdc/jjkim/NL2QGM/data/samsung/tables.json'
        with open(tables_file, 'r') as f:
            tables = json.load(f)
        db = 'dbname=samsung user=postgres password=postgres host=localhost port=5435'
        schema = Schema(get_schema(db, db_type='postgresql'), tables=tables[0])
        sql = get_sql(schema, query)
        print(sql)

        return sql

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
        self._test_query(query)
        

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
        self._test_query(query)

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
        self._test_query(query)

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
        self._test_query(query)

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
        self._test_query(query)

    def test_current_timestamp_1(self):
        query = """
            select *
            from master.m_defect_item
            where line_id = 'KFBG'
            and process_id = 'KGKV'
            and last_update_time > current_timestamp + '-7 days'
        """
        self._test_query(query)

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
        self._test_query(query)


if __name__ == '__main__':
    unittest.main()