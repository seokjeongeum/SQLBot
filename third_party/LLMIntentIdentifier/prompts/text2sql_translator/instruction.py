text2sql_translator_instruction_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>Using f_sql() API, return the SQL query that corresponds to the user's natural language query. 
Strictly follow the format of the below examples. 

{few_shot_examples}

<|eot_id|><|start_header_id|>user<|end_header_id|>
user: {question}
schema: {schema}
sql: <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""