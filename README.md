# SQLBot
```bash
python3.10 -m pip install hydra-core
python3.10 -m pip install -r SQLBot/requirements.txt

cd SQLBot/third_party/nl2qgm
export PYTHONPATH=.:ratsql/resources
CUDA_VISIBLE_DEVICES=1 python3.10 demo/backend_server.py

python3.10 SQLBot/third_party/LLMIntentIdentifier/intent_identifying.py
```