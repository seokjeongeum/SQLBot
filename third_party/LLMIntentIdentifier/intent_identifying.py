import logging
import argparse
from typing import Dict
import torch
import hydra
from flask import Flask, request
from flask_cors import CORS
from waitress import serve
from llm_based_intent_identifier import IntentIdentifier

# Initialize argument parser
# parser = argparse.ArgumentParser(description="Run the Flask app with custom index and ids paths")
# parser.add_argument('--edge_dataset_path', type=str, required=True, help='Path to edge dataset')

# args = parser.parse_args()

# Initialize Flask
app = Flask(__name__)
cors = CORS(app)

hydra.initialize(config_path="conf")
cfg = hydra.compose(config_name="intent_identifier")

intent_identifier = IntentIdentifier(cfg)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    return {"intent": intent_identifier.predict(data['question'])}

if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    serve(app, host="0.0.0.0", port=5000)