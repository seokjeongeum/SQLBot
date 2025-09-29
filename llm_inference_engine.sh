export CUDA_VISIBLE_DEVICES=0
docker run --gpus all \
    -p 30000:30000 \
    --env "HF_TOKEN=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" \
    --ipc=host \
    --mount type=bind,source=/mnt,target=/mnt \
    lmsysorg/sglang:latest \
    python3 -m sglang.launch_server --model-path /mnt/sdb1/shpark/meta-llama-3.1-8B-instruct --host 0.0.0.0 --port 30000 --tp 1