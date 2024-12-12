export CUDA_VISIBLE_DEVICES=0
docker run --gpus all \
    -p 30000:30000 \
    --env "HF_TOKEN=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" \
    --ipc=host \
    --mount type=bind,source=/mnt,target=/mnt \
    lmsysorg/sglang:latest \
    python3 -m sglang.launch_server --model-path /mnt/md0/jeseok/Llama-3.1-8B --host 0.0.0.0 --port 30000 --tp 1
    
    # --dp 1
# /mnt/sde/shpark/models/mistralai/Ministral-8B-Instruct-2410 (75, 150, 200), # (50, 100)
# /mnt/sde/shpark/models/google/gemma-2-9b-it (75, 150, 200), # (50, 100)
# /mnt/sde/shpark/models/Qwen/Qwen2.5-7B-Instruct #(75, 150, 200), # (50, 100)
# --kv-cache-dtype fp8_e5m2 --context-length 6400 