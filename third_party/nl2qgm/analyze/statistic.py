import os
import re
import json
import tqdm

result_data_path = "/mnt/nfs-server/backup/20210429/glove_run_no_Join_cond_true_1-step39200-eval.json"
inferred_data_path = "/mnt/nfs-server/backup/20210429/glove_run_no_Join_cond_true_1-step39200-infer.jsonl"
g_cnt = 0

WHERE_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists', 'notlike')

def load_jsonl(file_path):
    with open(file_path) as f:
        return [json.loads(line) for line in f.readlines()]

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def remove_table_name(clause):
    tmp = []
    for tok in clause.split(' '):
        if '.' in tok:
            tok_s = tok.split('.')
            if 'table' in tok_s[0] and 'col' in tok_s[1]:
                if '(' in tok_s[0]:
                    tok = tok_s[0].split('(')[0] + '(' + tok_s[1].upper()
                else:
                    tok = tok_s[1].upper()
        tmp.append(tok)
    return ' '.join(tmp)

def remove_values_(sql):
    tmp = sql.split(' WHERE ')
    if len(tmp) == 1:
        return sql
    elif len(tmp) == 2:
        s_quo_cnt = False
        d_quo_cnt = False
        condition_toks = []
        for tok in tmp[1].split(' '):
            if '"' in tok and tok.count('"') % 2:
                d_quo_cnt = not d_quo_cnt
                if not d_quo_cnt:
                    tok = "'terminal'"
            if "'" in tok and tok.count("'") % 2:
                s_quo_cnt = not s_quo_cnt
                if not s_quo_cnt:
                    tok = "'terminal'"
            if tok.count('"') == 2 or tok.count("'") == 2:
                tok = "'terminal'"
            # Append
            if d_quo_cnt or s_quo_cnt:
                continue
            else:
                condition_toks.append(tok)
        tmp[1] = ' '.join(condition_toks)
        # where_clause = re.sub(r"\"[\w\s]*\"", "\"terminal\"", tmp[1])
        # tmp[1] = where_clause
        return ' WHERE '.join(tmp)
    else:
        global g_cnt
        g_cnt += 1
        return sql

def remove_values(sql):
    s_quo_cnt = False
    d_quo_cnt = False
    new_sql = []
    for idx, letter in enumerate(sql):
        if letter == '"':
            d_quo_cnt = not d_quo_cnt
            if not d_quo_cnt:
                letter = [l for l in "'terminal'"]
        elif letter == "'":
            s_quo_cnt = not s_quo_cnt
            if not s_quo_cnt:
                letter = [l for l in "'terminal'"]
        # Add
        if not s_quo_cnt and not d_quo_cnt:
            if len(letter) == 1:
                new_sql.append(letter)
            else:
                new_sql += letter
    return ''.join(new_sql)

            
        



def remove_distinct(sql):
    return sql.replace(' DISTINCT ', ' ')

def upper_agg_funcs(sql):
    agg_funcs = ['Count(', 'Sum(', 'Max(', 'Min(', 'Avg(']
    for func in agg_funcs:
        sql = sql.replace(func, func.upper())
    return sql

def upper_key_words(sql):
    keywords = ['Desc', 'Asc', 'Limit', 'desc', 'asc', 'limit']
    for keyword in keywords:
        sql = sql.replace(f' {keyword} ', f' {keyword} '.upper())
    return sql


# SELECT
def get_select_clause(sql):
    return sql.split(" FROM ")[0]

def compare_select_clause(pred, gold):
    p = get_select_clause(pred)
    g = get_select_clause(gold)
    return not p == g


def get_agg_funcs(sql):
    agg_funcs = ['COUNT', 'SUM', 'MAX', 'MIN', 'AVG']
    tmp = []
    for tok in sql.split(' '):
        for func in agg_funcs:
            if f'{func}(' in tok.upper():
                tmp.append(func)
                break
    return tmp

def get_cols_(sql):
    regex = re.compile(r'\d\d\d-\d\d\d-\d\d\d\d')
    regex = re.compile(r'COL[\d]*')
    matchobj = regex.search(sql)
    if matchobj:
        cols = matchobj.group()
    else:
        cols = []
    return cols

def get_cols(sql):
    cols = []
    for idx, tok in enumerate(sql):
        if idx+3 < len(sql) and sql[idx:idx+3] == 'COL' and is_number(sql[idx+3]):
            end_idx = idx+3
            while end_idx < len(sql):
                if not is_number(sql[end_idx]):
                    break
                end_idx += 1
            cols.append(sql[idx:end_idx])
        elif tok == '*':
            cols.append('*')
    return cols

def get_direction(clause):
    if 'DESC' in clause:
        return 'DESC'
    elif 'ASC' in clause:
        return 'ASC'
    else:
        return None

def remove_space(sql):
    sql = sql.replace('( ', '(')
    sql = sql.replace(' )', ')')
    return sql

def remove_double_space(sql):
    sql = sql.replace('  ', ' ')
    return sql.replace('  ', ' ')

def remove_last_space(sql):
    if sql[-1] == ' ':
        sql = sql[:-1]
    return sql

def get_num(sql):
    return len(sql.split(' '))

def concat_operator(sql):
    sql = sql.replace(" NOT LIKE ", " NOTLIKE ")
    return sql


# WHERE
def get_where_clause(sql):
    # s = sql.split(" WHERE ")
    s = split_with_parenthesis_by_toks(sql, "WHERE", True)
    return s[1] if len(s) > 1 else ''

def compare_where_condition(pred, gold):
    p = get_where_clause(pred)
    g = get_where_clause(gold)
    return not p == g

def split_with_parenthesis_by_toks(string, deliminator=' ', is_where=False):
    string_s = string.split(' ')
    p_cnt = 0
    split_indices = [-1]
    for idx, tok in enumerate(string_s):
        if tok == deliminator and p_cnt == 0:
            split_indices.append(idx)
        if '(' in tok:
            p_cnt += tok.count('(')
        if ')' in tok:
            p_cnt -= tok.count(')')
        if is_where and p_cnt == 0 and tok in ['ORDER', 'GROUP', 'LIMIT']:
            split_indices.append(idx)
            break
    if idx+1 == len(string_s):
        split_indices.append(idx+1)

    assert p_cnt == 0, "Bad parenthesis"

    # Split
    tmp = []
    for idx in range(len(split_indices)-1):
        start_idx = split_indices[idx] + 1
        end_idx = split_indices[idx+1]
        tmp.append(' '.join(string_s[start_idx:end_idx]))
    return tmp

def split_with_parenthesis(string, deliminator=' '):
    p_cnt = 0
    split_indices = [-1]
    for idx, letter in enumerate(string):
        if letter == deliminator and p_cnt == 0:
            split_indices.append(idx)
        if letter == '(':
            p_cnt += 1
        elif letter == ')':
            p_cnt -= 1
    split_indices.append(idx+1)

    assert p_cnt == 0, "Bad parenthesis"

    # Split
    tmp = []
    for idx in range(len(split_indices)-1):
        start_idx = split_indices[idx] + 1
        end_idx = split_indices[idx+1]
        tmp.append(string[start_idx:end_idx])
    return tmp


def get_conditions(sql):
    if not sql:
        return []
    brac_cnt = 0
    split_indices = [-1]
    sql_s = split_with_parenthesis(sql)
    for idx, tok in enumerate(sql_s):
        if '(' in tok:
            brac_cnt += 1
        if ')' in tok:
            brac_cnt -= 1
        if brac_cnt == 0 and tok in ['AND', 'OR']:
            split_indices.append(idx)
        if brac_cnt == 0 and (sql_s[idx:idx+2] == ['ORDER', 'BY'] or sql_s[idx:idx+2] == ['GROUP', 'BY'] or sql_s[idx] == 'LIMIT'):
            split_indices.append(idx)
            break
    if idx+1 == len(sql_s):
        split_indices.append(idx+1)
    conds = []
    for idx in range(len(split_indices)-1):
        start_idx = split_indices[idx]
        end_idx = split_indices[idx+1]
        conds.append(sql_s[start_idx+1:end_idx])
    return conds


def get_operators(conds):
    ops = []
    for cond in conds:
        for tok in cond:
            if tok.lower() in WHERE_OPS:
                ops.append(tok)
            elif 'COL' not in tok:
                if tok not in ['*', '+', '-', "'terminal'", '"terminal"'] and not is_number(tok) and not (tok[0] == '(' and tok[-1] == ')'):
                    raise RuntimeError("Should not be here")
    return ops

def get_values(conds):
    values = []
    for cond in conds:
        for tok in cond:
            if '(SELECT' in tok or tok in ['"terminal"', "'terminal'", 'terminal']:
                values.append(tok)
    return values


def get_from_clause(sql):
    sql_s = sql.split(' ')
    start_idx = 0
    end_idx = 0
    for idx, tok in enumerate(sql_s):
        if tok == 'FROM':
            start_idx = idx
        if start_idx and tok in ['WHERE', 'ORDER', 'GROUP', 'LIMIT']:
            end_idx = idx
            break
    assert start_idx != 0 
    if end_idx == 0:
        end_idx = len(sql_s)

    return ' '.join(sql_s[start_idx:end_idx]).upper()

def compare_from_condition(pred, gold):
    p = get_from_clause(pred)
    g = get_from_clause(gold)
    return not p == g


def get_order_by_clause(sql):
    b_cnt = 0
    start_idx = 0
    end_idx = 0
    sql_s = sql.split(' ')
    for idx, tok in enumerate(sql_s):
        if '(' in tok:
            b_cnt += 1
        if ')' in tok:
            b_cnt -= 1
        # Found start_idx
        if b_cnt == 0 and tok == 'ORDER' and idx+1 != len(sql_s) and sql_s[idx+1] == 'BY':
            start_idx = idx
            if (idx+3) < len(sql_s) and sql_s[idx+3] in ['DESC', 'ASC']:
                end_idx = idx+4
            else:
                end_idx = idx+3
    if start_idx != 0:
        return ' '.join(sql_s[start_idx:end_idx])
    else:
        return ''

def get_group_by_clause(sql):
    b_cnt = 0
    start_idx = 0
    end_idx = 0
    sql_s = sql.split(' ')
    for idx, tok in enumerate(sql_s):
        if '(' in tok:
            b_cnt += 1
        if ')' in tok:
            b_cnt -= 1
        # Found start_idx
        if b_cnt == 0 and tok == 'GROUP' and idx+1 != len(sql_s) and sql_s[idx+1] == 'BY':
            start_idx = idx
        if start_idx and 'COL' in tok:
            end_idx = idx+1
        elif start_idx and tok in ['ORDER', 'LIMIT', 'HAVING']:
            break
    
    if start_idx != 0:
        return ' '.join(sql_s[start_idx:end_idx])
    else:
        return ''    


def compare_group_by(pred, gold):
    p = get_group_by_clause(pred)
    g = get_group_by_clause(gold)
    return not p == g


def compare_order_by(pred, gold):
    p = get_order_by_clause(pred)
    g = get_order_by_clause(gold)
    return not p == g


def get_limit(sql):
    b_cnt = 0
    sql_s = sql.split(' ')
    for idx, tok in enumerate(sql_s):
        if '(' in tok:
            b_cnt += 1
        if ')' in tok:
            b_cnt -= 1
        if b_cnt == 0 and tok == 'LIMIT':
            assert idx+1 < len(sql_s)
            return sql_s[idx+1]
    return None

def compare_limit(pred, gold):
    p = get_limit(pred)
    g = get_limit(gold)
    return not p == g


def remove_redundant_parenthesis(sql):
    toks = []
    for tok in sql.split(' '):
        if 'COL' in tok and tok[0] == '(' and tok[-1] == ')' and len(tok) < 7:
            toks.append(tok[1:-1])
        else:
            toks.append(tok)
    return ' '.join(toks)

def has_subquery(sql):
    return " (SELECT " in sql or '( SELECT' in sql

def has_where(sql):
    return ' WHERE ' in sql

def has_orderby(sql):
    sql_s = sql.split(' ')
    b_cnt = 0
    for idx, tok in enumerate(sql_s):
        if '(' in tok:
            b_cnt += tok.count('(')
        if ')' in tok:
            b_cnt -= tok.count(')')
        if b_cnt == 0 and tok == 'ORDER' and sql_s[idx+1] == 'BY':
            return True
    return False


def has_group_by(sql):
    sql_s = sql.split(' ')
    b_cnt = 0
    for idx, tok in enumerate(sql_s):
        if '(' in tok:
            b_cnt += tok.count('(')
        if ')' in tok:
            b_cnt -= tok.count(')')
        if b_cnt == 0 and tok == 'GROUP' and sql_s[idx+1] == 'BY':
            return True
    return False
    

def has_limit(sql):
    sql_s = sql.split(' ')
    b_cnt = 0
    for idx, tok in enumerate(sql_s):
        if '(' in tok:
            b_cnt += tok.count('(')
        if ')' in tok:
            b_cnt -= tok.count(')')
        if b_cnt == 0 and tok == 'limit' and is_number(sql_s[idx+1]):
            return True
    return False

def has_arth_op(clause):
    ar_ops = ['+', '-', '/']
    for op in ar_ops:
        if op in clause:
            return True
    return False


if __name__ == "__main__":
    infer_data = load_jsonl(inferred_data_path)
    data = json.load(open(result_data_path))['per_item']

    assert len(data) == len(infer_data)
    tmp = []
    for datum, infer_datum in zip(data, infer_data):
        if 'db_id' not in datum:
            datum['db_id'] = infer_datum['db_id']
        datum['question'] = infer_datum['beams'][0]['orig_question']
        datum['predicted'] = remove_table_name(datum['predicted'])
        datum['predicted'] = upper_agg_funcs(datum['predicted'])
        datum['predicted'] = upper_key_words(datum['predicted'])
        datum['predicted'] = remove_distinct(datum['predicted'])
        datum['predicted'] = concat_operator(datum['predicted'])
        datum['gold'] = remove_values(datum['gold'])
        datum['gold'] = remove_space(datum['gold'])
        datum['gold'] = remove_distinct(datum['gold'])
        datum['gold'] = concat_operator(datum['gold'])
        datum['gold'] = remove_double_space(datum['gold'])
        datum['gold'] = remove_last_space(datum['gold'])
        datum['gold'] = remove_redundant_parenthesis(datum['gold'])

    # Count correct and wrong
    correct_data = list(filter(lambda k: k['exact'], data))
    wrong_data = list(filter(lambda k: not k['exact'], data))

    wrong_select_column = []
    wrong_where_column = []
    wrong_group_by_column = []
    wrong_orderby_column = []
    wrong_from = []
    
    # SELECT
    wrong_select = []
    wrong_select_arthmetic_op = []
    wrong_select_num = []
    wrong_select_agg = []

    # WHERE
    wrong_where = []
    wrong_cond_num = []
    wrong_ops = []
    wrong_values = []

    # GROUP BY
    wrong_group_by_existence = []
    wrong_group_by_num = []
    wrong_group_by_agg = []

    # ORDER BY
    wrong_orderby_existence = []
    wrong_orderby_direction = []

    # Limit
    wrong_limit = []

    for idx, datum in enumerate(tqdm.tqdm(wrong_data)):
        pred, gold = datum['predicted'], datum['gold']
        if compare_select_clause(pred, gold):
            p, g = get_select_clause(pred), get_select_clause(gold)
            p_num, g_num = get_num(p), get_num(g)
            p_aggs, g_aggs = get_agg_funcs(p), get_agg_funcs(g)
            p_cols, g_cols = get_cols(p), get_cols(g)
            # If num
            if p_num != g_num and (not has_arth_op(p) and not has_arth_op(g)):
                wrong_select_num.append(datum)
            # If agg
            elif p_aggs != g_aggs:
                wrong_select_agg.append(datum)
            elif p_num != g_num:
                wrong_select_arthmetic_op.append(datum)
            # If Col
            elif p_cols != g_cols:
                wrong_select_column.append(datum)
            else:
                raise RuntimeError("Should not be here")
            wrong_select.append(datum)
        elif compare_where_condition(pred, gold):
            p, g = get_where_clause(pred), get_where_clause(gold)
            p_conds, g_conds = get_conditions(p), get_conditions(g)
            p_cols, g_cols = get_cols(p), get_cols(g)
            p_ops, g_ops = get_operators(p_conds), get_operators(g_conds)
            p_values, g_values = get_values(p_conds), get_values(g_conds)
            # Number different
            if len(p_conds) != len(g_conds):
                wrong_cond_num.append(datum)
            
            # Operator different
            elif p_ops != g_ops:
                wrong_ops.append(datum)

            # column different
            elif p_cols != g_cols:
                wrong_where_column.append(datum)
            
            # Nested query different
            elif p_values != g_values:
                wrong_values.append(datum)
            else:
                raise RuntimeError("Should not be here")
            wrong_where.append(datum)
        
        elif compare_from_condition(pred, gold):
            wrong_from.append(datum)

        elif compare_group_by(pred, gold):
            p, g = get_group_by_clause(pred), get_group_by_clause(gold)
            p_cols, g_cols = get_cols(p), get_cols(g)
            # p_aggs, g_aggs = get_agg_funcs(p), get_agg_funcs(g)

            if len(p_cols) != len(g_cols):
                if p == '' or g == '':
                    wrong_group_by_existence.append(datum)
                else:
                    wrong_group_by_num.append(datum)
            elif p_cols != g_cols:
                wrong_group_by_column.append(datum)
            else:
                raise RuntimeError("Should not be here")

        elif compare_order_by(pred, gold):
            p, g = get_order_by_clause(pred), get_order_by_clause(gold)
            p_cols, g_cols = get_cols(p), get_cols(g)
            p_direction, g_direction = get_direction(p), get_direction(g)
            assert len(p_cols) == 1 or len(g_cols) == 1
            if p_cols != g_cols:
                if len(p_cols) == 0 or len(g_cols) == 0:
                    wrong_orderby_existence.append(datum)
                else:
                    wrong_orderby_column.append(datum)
            elif p_direction != g_direction:
                wrong_orderby_direction.append(datum)
            else:
                raise RuntimeError("Should not be here")
        elif compare_limit(pred, gold):
            p, g = get_limit(pred), get_limit(gold)
            wrong_limit.append(datum)
        else:
            raise RuntimeError("Should not be here")


    ## SELECT 
    cnt = 0
    # wrong_select_num
    print(f"Wrong select num:{len(wrong_select_num)}")
    cnt += len(wrong_select_num)
    # wrong_select_arthmetic_op
    print(f"Wrong select arth op:{len(wrong_select_arthmetic_op)}")
    cnt += len(wrong_select_arthmetic_op)
    # wrong_select_agg
    print(f"Wrong select agg:{len(wrong_select_agg)}")
    cnt += len(wrong_select_agg)
    # wrong_select_column
    print(f"Wrong select col:{len(wrong_select_column)}")
    cnt += len(wrong_select_column)

    ## WHERE
    # wrong_cond_num
    print(f"Wrong where condition num:{len(wrong_cond_num)}")
    cnt += len(wrong_cond_num)
    # wrong_ops
    print(f"Wrong where ops:{len(wrong_ops)}")
    cnt += len(wrong_ops)
    # Wrong column
    print(f"Wrong where column:{len(wrong_where_column)}")
    cnt += len(wrong_where_column)
    # wrong_values
    print(f"Wrong where values:{len(wrong_values)}")
    cnt += len(wrong_values)

    ## GROUP BY
    # wrong_group_by_existence
    print(f"Wrong group by existence:{len(wrong_group_by_existence)}")
    cnt += len(wrong_group_by_existence)
    # Wrong_group_by_num
    print(f"Wrong group by num:{len(wrong_group_by_num)}")
    cnt += len(wrong_group_by_num)
    # Wrong group by agg
    print(f"Wrong group by aggs:{len(wrong_group_by_agg)}")
    cnt += len(wrong_group_by_agg)
    # wrong_group_by_column
    print(f"Wrong group by column:{len(wrong_group_by_column)}")
    cnt += len(wrong_group_by_column)

    ## ORDER BY
    # wrong_ordeby_existence
    print(f"Wrong order by existence:{len(wrong_orderby_existence)}")
    cnt += len(wrong_orderby_existence)
    # wrong_orderby_column
    print(f"Wrong order by column:{len(wrong_orderby_column)}")
    cnt += len(wrong_orderby_column)
    # wrong_orderby_direction
    print(f"Wrong order by direction:{len(wrong_orderby_direction)}")
    cnt += len(wrong_orderby_direction)

    # LIMIT
    # wrong_limit
    print(f"Wrong limit:{len(wrong_limit)}")
    cnt += len(wrong_limit)
    
    print(f"Wrong from:{len(wrong_from)}")
    cnt += len(wrong_from)

    stop = 1

    # Count where:  (wrong_gold / total_gold)
    where_in_wrong = list(filter(lambda k: has_where(k['gold']), wrong_data))
    where_in_total = list(filter(lambda k: has_where(k['gold']), data))
    print(f"Where: {len(where_in_wrong)/len(where_in_total)} ({len(where_in_wrong)}/{len(where_in_total)})")
    # Count group by:
    group_by_in_wrong = list(filter(lambda k: has_group_by(k['gold']), wrong_data))
    group_by_in_total = list(filter(lambda k: has_group_by(k['gold']), data))
    print(f"Group by:{len(group_by_in_wrong)/len(group_by_in_total)} ({len(group_by_in_wrong)}/{len(group_by_in_total)})")
    # Count order by
    order_by_in_wrong = list(filter(lambda k: has_orderby(k['gold']), wrong_data))
    order_by_in_total = list(filter(lambda k: has_orderby(k['gold']), data))
    print(f"Order by:{len(order_by_in_wrong)/len(order_by_in_total)} ({len(order_by_in_wrong)}/{len(order_by_in_total)})")
    # Count nested query
    sub_query_in_wrong = list(filter(lambda k: has_subquery(k['gold']), wrong_data))
    sub_query_in_total = list(filter(lambda k: has_subquery(k['gold']), data))
    print(f"Subquery by:{len(sub_query_in_wrong)/len(sub_query_in_total)} ({len(sub_query_in_wrong)}/{len(sub_query_in_total)})")
    
    stop = 1


    # Show 
    for datum in wrong_select_num:
        print(f"\nNL: {datum['question']}")
        print(f"PRED: {datum['predicted']}")
        print(f"GOLD: {datum['gold']}")

    stop = 1
    # Show by level: