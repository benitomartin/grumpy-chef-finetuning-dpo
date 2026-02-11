

# Lower memory utilization

vllm serve benitomartin/grumpy-chef-lfm2.5-1.2B-vllm --dtype bfloat16 --gpu-memory-utilization 0.7 --max-model-len 512


curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "benitomartin/grumpy-chef-lfm2.5-1.2B-vllm",
    "messages": [{"role": "user", "content": "Can I put cream in carbonara?"}],
    "max_tokens": 100,
    "temperature": 0.3
  }'