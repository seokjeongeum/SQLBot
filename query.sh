curl -X POST http://141.223.199.1:7000/text_to_sql \
-H "Content-Type: application/json" \
-d '{"text": "how many singer?", "db_id": "concert_singer", "analyse": "true", "reset_history": "false"}'
