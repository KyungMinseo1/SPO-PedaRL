#!/usr/bin/env bash
./stop_vllm_server.sh

# Explicitly pass through environment variables
export CUDA_VISIBLE_DEVICES
python vllm_server.py "$@"
