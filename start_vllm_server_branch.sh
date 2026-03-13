#!/usr/bin/env bash
./stop_vllm_server_branch.sh

python vllm_server_branch.py "$@"
