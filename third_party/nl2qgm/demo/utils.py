from typing import *

import en_core_web_trf
import psycopg2
import spacy

model = spacy.load("en_core_web_trf")
model = en_core_web_trf.load()


def extract_nouns(
    sentence: str, enable_PROPN: bool = False, model: spacy.Language = model
) -> List[str]:
    # Define target POS tags
    target_pos = ["PROPN", "NOUN"] if enable_PROPN else ["NOUN"]

    # Perform parsing
    parsed_doc = model(sentence)

    print(sentence)

    # Extract nouns
    flag = False
    nouns = []
    tmp_word = []
    for word in parsed_doc:
        w_text = word.text
        w_pos = word.pos_
        # If the word is a noun
        if w_pos in target_pos:
            if flag == False:
                # If it is the beginning of a noun phrase
                tmp_word.append(w_text)
                flag = True
            else:
                # Add the word to the noun phrase (if it is not the beginning)
                tmp_word.append(w_text)
        else:
            if flag == True:
                # End of noun phrase
                nouns.append(" ".join(tmp_word))
                tmp_word = []
                flag = False

    if flag:
        nouns.append(" ".join(tmp_word))
    print(nouns)

    # Debugging
    print([(word.text, word.pos_) for word in parsed_doc])

    return nouns


def token_score_to_noun_score(
    tokens: List[str], token_scores: List[float]
) -> Tuple[List[str], List[int]]:
    words, word_scores = token_score_to_word_score(tokens, token_scores)

    nouns = extract_nouns(sentence=" ".join(words), enable_PROPN=True)

    # Extract only noun words
    filtered_indices = []
    filtered_words = []
    filtered_scores = []
    for idx, (word, score) in enumerate(zip(words, word_scores)):
        if any([word in noun for noun in nouns]):
            filtered_indices.append(idx)
            filtered_words.append(word)
            filtered_scores.append(score)
    # Combine the word and score if they are consecutive
    combined_words: List[List[str]] = []
    combined_scores: List[List[float]] = []
    for idx, word, score in zip(filtered_indices, filtered_words, filtered_scores):
        if idx - 1 in filtered_indices:
            combined_words[-1].append(word)
            combined_scores[-1].append(score)
        else:
            combined_words.append([word])
            combined_scores.append([score])
    # Combine
    filtered_words = [" ".join(words) for words in combined_words]
    filtered_scores = [sum(scores) / len(scores) for scores in combined_scores]

    return filtered_words, filtered_scores


def token_score_to_word_score(
    tokens: List[str], token_scores: List[float]
) -> Tuple[List[str], List[float]]:
    assert len(tokens) == len(
        token_scores
    ), f"Length of tokens and token_scores must be equal. Got {len(tokens)} and {len(token_scores)} respectively."

    score_buf = []
    word_buf = []
    final_scores = []
    final_words = []
    for idx, token in enumerate(tokens):
        if token.startswith("##"):
            word_buf.append(token[2:])
            score_buf.append(token_scores[idx])
        elif len(word_buf) == 0:
            word_buf.append(token)
            score_buf.append(token_scores[idx])
        else:
            # Move buf to final
            # Average the scores
            word_score = sum(score_buf) / len(score_buf)
            word = "".join(word_buf)
            # Add to final scores
            final_scores.append(word_score)
            final_words.append(word)
            # Add current token to buf
            word_buf = [token]
            score_buf = [token_scores[idx]]

    # Add the last word
    if tokens:
        word_score = sum(score_buf) / len(score_buf)
        word = "".join(word_buf)
        final_scores.append(word_score)
        final_words.append(word)

    return final_words, final_scores


def all_values_from_db(db_name: str, table_name: str, column_name: str) -> List[str]:
    # Connect to DB
    pg_config = (
        f"host=localhost port=5434 user=sqlbot password=sqlbot_pw dbname={db_name}"
    )
    # Find value from DB (For string values)
    with psycopg2.connect(pg_config) as conn:
        with conn.cursor() as cursor:
            search_query = f"SELECT {column_name} FROM {table_name}"
            cursor.execute(search_query)
            results = cursor.fetchall()
            if not results:
                return []
            return [str(result[0]) for result in results]


def add_value_one_sql(question: str, db_name: str, sql: str, history: str) -> str:
    """Assumption: There are no repeated values in the question."""
    # Parse history
    history_list = history.lower().split("<s>")

    question = question.lower()

    flag = False
    target_text = question
    while "'terminal'" in sql:
        terminal_start_idx = sql.index("'terminal'")
        terminal_end_idx = terminal_start_idx + len("'terminal'")
        if flag:
            if len(history_list) == 0:
                break
            target_text = history_list.pop()

        # Find the table.column for the terminal
        found_flag = False
        # Find table and column name
        tab_col = sql[:terminal_start_idx].strip().split(" ")[-2]
        table, column = tab_col.split(".")
        # Find all possible values for the column
        values = all_values_from_db(db_name, table, column)
        # Check if any of the values are in the question
        for value in values:
            if value.lower() in target_text:
                # Replace terminal with value
                front_sub_sql = sql[:terminal_start_idx]
                back_sub_sql = sql[terminal_end_idx:]
                sql = front_sub_sql + f"'{value}'" + back_sub_sql

                # Find the word position in the question and remove it (Remove only the first occurrence)
                start_idx = target_text.index(value.lower())
                end_idx = start_idx + len(value)
                target_text = target_text[:start_idx] + target_text[end_idx:]
                target_text = target_text.replace("  ", " ")
                found_flag = True
                break

        if not found_flag:
            flag = True

    return sql


def infer_value_from_question(question, table, column, db, infer_value_cnt):
    def increase_value_cnt():
        infer_value_cnt[0] += 1

    def is_int(word):
        try:
            int(word)
            return True
        except:
            return False

    def sent_to_words(sent):
        words = sent.replace(".", "").replace("  ", " ").split(" ")
        return words

    pg_config = "host=localhost port=5434 user=postgres password=postgres dbname=" + db
    # Try to find number values from string
    words = sent_to_words(question)
    values = [word for word in words if is_int(word)]
    if values:
        if len(values) <= infer_value_cnt[0]:
            return values[-1]
        else:
            tmp = infer_value_cnt[0]
            increase_value_cnt()
            return values[tmp]
    # Find value from DB (For string values)
    with psycopg2.connect(pg_config) as conn:
        with conn.cursor() as cursor:
            search_query = f"SELECT {column} FROM {table}"
            cursor.execute(search_query)
            results = cursor.fetchall()
            if not results:
                return "value_not_found"
            for result in results:
                if str(result[0]) in question:
                    return result[0]

            return result[0][0]
