#!/usr/bin/env bash
pkill -f uvicorn || true
pkill -f multiprocess.spawn || true
pkill -f vllm_server_branch.py || true