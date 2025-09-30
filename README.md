# SQLBot

## Setup
```bash
# Create and activate conda environment
conda create -n sqlbot python=3.10 -y
conda activate sqlbot

# Install system dependencies using conda
conda install -c conda-forge postgresql libpq cairo pygobject dbus -y

# Install Python dependencies
pip install -r requirements.txt
```

## LLM Inference Engine
```bash
# Start LLM inference server (runs on port 30000)
./llm_inference_engine.sh
```

## Backend Server
```bash
cd third_party/nl2qgm
export PYTHONPATH=.:ratsql/resources
CUDA_VISIBLE_DEVICES=1 python demo/backend_server.py
```

## Intent Identifier
```bash
python third_party/LLMIntentIdentifier/intent_identifying.py
```

## Frontend
```bash
cd web

# Install dependencies
npm install

# Start development server (opens at http://localhost:3000)
npm run dev
```

## API Endpoints

The Backend Server exposes the following endpoints:

### `/table_to_text` (POST)
Converts SQL query results into natural language summaries.
- **Implementation**: `third_party/nl2qgm/table2text/`
  - `model.py` - Main table-to-text model class
  - `llama.py` - LLM API wrapper for text generation

### `/text_to_sql` (POST)
Translates natural language queries into SQL.
- **Implementation**: `third_party/nl2qgm/demo/backend_server.py`
- **Intent Classification**: `third_party/LLMIntentIdentifier/`
  - `llm_based_intent_identifier.py` - Intent identification logic
  - `intent_identifying.py` - Flask server for intent prediction (port 5000)
