# SQLBot
```bash
apt update
apt install libpq-dev python-dev -y
apt install libcairo2-dev -y
apt install libgirepository1.0-dev -y
apt install libdbus-1-dev -y

python3.10 -m pip install hydra-core
python3.10 -m pip install -r SQLBot/requirements.txt

cd SQLBot/third_party/nl2qgm
export PYTHONPATH=.:ratsql/resources
CUDA_VISIBLE_DEVICES=1 python3.10 demo/backend_server.py

python3.10 SQLBot/third_party/LLMIntentIdentifier/intent_identifying.py
```
