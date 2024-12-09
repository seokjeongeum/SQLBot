intent_identifier_instruction_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>Using f_tune() API, return True to detect when an user message involves performing tuning operations on the database. 
Strictly follow the format of the below examples. 

{few_shot_examples}

<|eot_id|><|start_header_id|>user<|end_header_id|>
user: {question}
intent: <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""