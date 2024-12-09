# Backend Server

## Setting
To run the Backend Server, a backend_config.json containing setting information is required. The setting information contained in the backend_config.json is broadly divided into server, model, and data. The location of the backend_config.json file should be placed in the demo folder.
- For the server, the information that must be written includes the host and port of the backend server, the host and port of the redis server used for caching, and information about the api_cache_db.
- For the model, it is composed of text_to_sql, text_to_intent, and result_analysis models. For each model, the location of the config.jsonnet file containing the configuration information used during model training, the location of the checkpoint, and device information must be specified.
```
text_to_sql: This model translates a user's natural language query into the corresponding SQL query.
text_to_intent: This model predicts the intent of a user's natural language query.
result_analysis: This model checks the impact of each token when a user's natural language query is tokenized on result confidence using the integrated gradient method. Although it uses the same model as the text_to_sql model, it uses the path of config_captum.jsonnet that includes additional information necessary for integrated gradients as the experiment_config_path.
```
- In the case of data, it is based on the spider dataset and requires specifying the location of the spider database directory and the location of tables.json.
## Run the Server
The backend server can be run through the following command line.
```
cd /nl2qgm/demo
python ./backend_server.py
```
The backend server must receive the following input dictionary from the client.
* text (str): The user's NL (natural language) question
* db_id (str): The name of the database
* analyse (bool): Whether to perform integrated gradient analysis