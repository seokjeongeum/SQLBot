from ratsql.datasets.utils import db_utils


def insert():
    db_conn_str = 'user=postgres password=postgres host=localhost port=5435 dbname=samsung'
    with db_utils.connect(db_conn_str, "postgres") as conn:
        cursor = conn.cursor()
        cursor.execute("ALTER TABLE quality.m_fab_tracking ALTER COLUMN root_lot_id TYPE VARCHAR (30);")
        cursor.execute("ALTER TABLE quality.m_defect_chip ALTER COLUMN part_id TYPE VARCHAR (30);")
        cursor.execute("ALTER TABLE quality.m_fab_dcop_met ALTER COLUMN part_id TYPE VARCHAR (30);")
        cursor.execute("ALTER TABLE quality.m_eds_chip_bin ALTER COLUMN wafer_id TYPE VARCHAR (30);")
        cursor.execute("ALTER TABLE quality.m_eds_chip_bin ALTER COLUMN last_update_time DROP NOT NULL;")
        cursor.execute("ALTER TABLE quality.m_eds_chip_bin ALTER COLUMN chip_x_pos DROP NOT NULL;")
        cursor.execute("ALTER TABLE quality.m_eds_chip_bin ALTER COLUMN chip_y_pos DROP NOT NULL;")
        cursor.execute("ALTER TABLE quality.m_eds_chip_bin ALTER COLUMN good_bin DROP NOT NULL;")
        cursor.execute("ALTER TABLE quality.m_eds_chip_bin ALTER COLUMN dut_no DROP NOT NULL;")
        cursor.execute("ALTER TABLE quality.m_fab_dcop_met ALTER COLUMN root_lot_id TYPE VARCHAR (30);")
        cursor.execute("ALTER TABLE quality.m_fab_dcop_met ALTER COLUMN lot_id TYPE VARCHAR (30);")
        cursor.execute("ALTER TABLE quality.m_fab_dcop_met ALTER COLUMN wafer_id TYPE VARCHAR (30);")
        cursor.executemany("INSERT INTO quality.m_fab_tracking (root_lot_id, lot_id, process_id) VALUES (%s, %s, %s);",
                                              (('MTE056', 'lot_100', 'aPPIT'),
                                               ('MTE056', 'lot_101', 'PaPITU'),
                                               ('MTE056', 'lot_102', 'AaAIT'),
                                               ('MTE056', 'lot_103', 'AcvAAA'),
                                               ('ASD', 'lot_104', 'PPcIT'),
                                               ('DD', 'lot_105', 'PPIzT'),
                                               ('CCC', 'lot_106', 'USDfT'),
                                               ))
        cursor.executemany("INSERT INTO quality.m_fab_tracking (root_lot_id, lot_id, process_id) VALUES (%s, %s, %s);",
                           (('root_lot_1', 'lot_1', 'PPIT'),
                            ('root_lot_1', 'lot_2', 'PPITU'),
                            ('root_lot_1', 'lot_3', 'PPIT'),
                            ('root_lot_1', 'lot_4', 'PPIT'),
                            ('root_lot_2', 'lot_5', 'AAIT'),
                            ('root_lot_2', 'lot_6', 'AAAA'),
                            ('root_lot_2', 'lot_7', 'PPIT'),
                            ('root_lot_2', 'lot_8', 'PPIT'),
                            ('root_lot_3', 'lot_9', 'USDT'),
                            ('root_lot_3', 'lot_10', 'USDT'),
                            ('root_lot_4', 'lot_11', 'PPIT'),
                            ('root_lot_4', 'lot_12', 'PPIT'),
                            ))
        cursor.executemany("INSERT INTO quality.m_defect_chip (part_id, lot_id, total_dft_cnt) VALUES (%s, %s, %s);",
                           (('KHAA84901B-GEL', 'lot_1', 11),
                            ('KHAA84901B-GEL', 'lot_1', 3),
                            ('KHAA84901B-GEL', 'lot_2', 1),
                            ('KHAA84901B-GEL', 'lot_2', 4),
                            ('KHAA84901B-GEL', 'lot_2', 25),
                            ('KHAA84901B-GEL', 'lot_3', 3),
                            ('KHAA84901B-GEL', 'lot_3', 7),
                            ('part_1', 'lot_8', 1),
                            ('part_2', 'lot_9', 2),
                            ('part_3', 'lot_10', 17),
                            ('part_4', 'lot_11', 4),
                            ('part_5', 'lot_12', 3),
                            ))
        cursor.executemany("INSERT INTO quality.m_eds_chip_bin (wafer_id, lot_id, bin_no, process_id, line_id, root_lot_id, step_seq, tkout_time) VALUES (%s, %s, %s, %s, %s, %s, %s, %s);",
                           (('wafer_1', 'lot_1', '{1,2,3}', 'KCAK', 'KFBG', 'a', 's', '1996-04-15'),
                            ('wafer_2', 'lot_2', '{1,2}', 'KCAK', 'KFBG', 'a', 's', '1996-04-15'),
                            ('wafer_3', 'lot_3', '{1,2,3,4}', 'KCAK', 'KFBG', 'a', 's', '1996-04-15'),
                            ('wafer_4', 'lot_4', '{1,2,3,4,5}', 'KCAK', 'KFBG', 'a', 's', '1996-04-15'),
                            ('wafer_5', 'lot_5', '{1,2,3,4,5,6}', 'KCAK', 'KFBG', 'a', 's', '1996-04-15'),
                            ('wafer_6', 'lot_6', '{1,2}', 'KCAK', 'KFBG', 'a', 's', '1996-04-15'),
                            ('wafer_7', 'lot_7', '{1}', 'KCAK', 'KFBG', 'a', 's', '1996-04-15'),
                            ('wafer_8', 'lot_8', '{1,2,3}', 'AAA', 'KFBG', 'a', 's', '1996-04-15'),
                            ('wafer_9', 'lot_9', '{1,2}', 'BBB', 'KFBG', 'a', 's', '1996-04-15'),
                            ('wafer_10', 'lot_10', '{1,2,3}', 'CCC', 'KFBG', 'a', 's', '1996-04-15'),
                            ('wafer_11', 'lot_11', '{1,2}', 'DDD', 'KFBG', 'a', 's', '1996-04-15'),
                            ('wafer_12', 'lot_12', '{1,2,3}', 'EEE', 'KFBG', 'a', 's', '1996-04-15'),
                            ))
        cursor.executemany("INSERT INTO master.m_defect_item (line_id, step_seq, item_id, process_id) VALUES (%s, %s, %s, %s);",
                           (('KFBH', 'step_1', 'item_1', 'KCAK'),
                            ('KFBH', 'step_2', 'item_2', 'KCAK'),
                            ('KFBH', 'step_3', 'item_3', 'KGKV'),
                            ('KFBH', 'step_4', 'item_4', 'KGKV'),
                            ('line_1', 'step_5', 'item_5', 'KGKV'),
                            ('line_2', 'step_6', 'item_6', 'KCAK'),
                            ('line_3', 'step_7', 'item_7', 'KCAK'),
                            ('line_4', 'step_8', 'item_8', 'KGKV'),
                            ('line_5', 'step_9', 'item_9', 'BBB'),
                            ('KFBH', 'step_10', 'item_10', 'KGKV'),
                            ('KFBH', 'step_11', 'item_11', 'KGKV'),
                            ('KFBH', 'step_12', 'item_12', 'EEE'),
                            ))
        cursor.executemany("INSERT INTO quality.m_fab_dcop_met (part_id, root_lot_id, lot_id, wafer_id, line_id, eqp_id, item_id, fab_value) VALUES (%s, %s, %s, %s, %s, %s, %s, %s);",
                           (('K9GDGD8U0D-CXT', 'CKEBB7', 'lot_1', 'wafer_1', 'KFBG', 'eqp_1', '{1,2,3}', '{1,2,3,4}'),
                            ('K9GDGD8U0D-CXT', 'CKEBB7', 'lot_2', 'wafer_2', 'KFBG', 'eqp_2', '{1,2,3}', '{1,2,3,4}'),
                            ('K9GDGD8U0D-CXT', 'CKEBB7', 'lot_3', 'wafer_3', 'KFBG', 'eqp_3', '{1,2,3}', '{1,2,3,4}'),
                            ('K9GDGD8U0D-CXT', 'CKEBB7', 'lot_4', 'wafer_4', 'AAA', 'eqp_4', '{1,2,3}', '{1,2,3,4}'),
                            ('K9GDGD8U0D-CXT', 'CKEBB7', 'lot_5', 'wafer_5', 'BBB', 'eqp_5', '{1,2,3}', '{1,2,3,4}'),
                            ('K9GDGD8U0D-CXT', 'GKG706', 'lot_6', 'wafer_6', 'CC', 'eqp_6', '{1,2,3}', '{1,2,3,4}'),
                            ('K9GDGD8U0D-CXT', 'GKG706', 'lot_7', 'wafer_7', 'KFBG', 'eqp_7', '{1,2,3}', '{1,2,3,4}'),
                            ('wafer_8', 'GKG706', 'lot_8', 'wafer_8', 'DDDD', 'eqp_8', '{1,2,3}', '{1,2,3,4}'),
                            ('wafer_9', 'GKG706', 'lot_9', 'wafer_9', 'EEE', 'eqp_9', '{1,2,3}', '{1,2,3,4}'),
                            ('wafer_10', 'GKG706', 'lot_10', 'wafer_10', 'KFBG', 'eqp_10', '{1,2,3}', '{1,2,3,4}'),
                            ('wafer_11', 'GKG706', 'lot_11', 'wafer_11', 'KFBG', 'eqp_11', '{1,2,3}', '{1,2,3,4}'),
                            ('wafer_12', 'GKG706', 'lot_12', 'wafer_12', 'KFBG', 'eqp_12', '{1,2,3}', '{1,2,3,4}'),
                            ))
        conn.commit()


def scan():
    db_conn_str = 'user=postgres password=postgres host=localhost port=5435 dbname=samsung'
    with db_utils.connect(db_conn_str, "postgres") as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT root_lot_id, lot_id, process_id FROM quality.m_fab_tracking")
        print(cursor.fetchall())
        cursor.execute("SELECT part_id, lot_id, total_dft_cnt FROM quality.m_defect_chip")
        print(cursor.fetchall())
        cursor.execute("SELECT wafer_id, lot_id, bin_no, process_id, line_id, root_lot_id, step_seq, tkout_time FROM quality.m_eds_chip_bin")
        print(cursor.fetchall())
        cursor.execute("SELECT line_id, step_seq, item_id, process_id FROM master.m_defect_item")
        print(cursor.fetchall())
        cursor.execute("SELECT part_id, root_lot_id, lot_id, wafer_id, line_id, eqp_id, item_id, fab_value FROM quality.m_fab_dcop_met")
        print(cursor.fetchall())


if __name__ == "__main__":
    insert()
    #scan()