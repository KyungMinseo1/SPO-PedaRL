#!/usr/bin/env bash
set -euo pipefail

START_TIME=$(date +%s)

echo "======================================"
echo "🚀 Eval Start : $(date)"
echo "======================================"

clear_gpu() {
    echo ""
    echo "🧹 Cleaning up lingering processes..."
    
    pkill -9 -f "VLLM" || true
    pkill -9 -f "vllm" || true
    sleep 5

    echo "💧 Clearing internal Torch cache..."
    python - <<'PY'
import gc
import torch
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print("Internal cache cleared")
PY

    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "Current GPU memory usage:"
        nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
    fi
    echo "--------------------------------------"
}

run_eval () {
    local name=$1
    local config=$2

    echo ""
    echo "▶️ Start $name : $(date)"
    python eval.py --config-name "$config"
    echo "✅ Done  $name : $(date)"
    clear_gpu
}

run_eval_original () {
    local name=$1
    local config=$2

    echo ""
    echo "▶️ Start $name : $(date)"
    python eval_original.py --config-name "$config"
    echo "✅ Done  $name : $(date)"
    clear_gpu
}

# run_eval "Qwen2.5-0.5B" "Qwen2.5-0.5B-Instruct.yaml"
run_eval "SocraticLM" "SocraticLM.yaml"
run_eval "TutorRL" "TutorRL.yaml"
run_eval "TutorRL-think" "TutorRL-think.yaml"
run_eval "Qwen2.5-7B" "Qwen2.5-7B-Instruct.yaml"
run_eval "ConstructivistRL" "ConstructivistRL-GDPO-7b_Soft0.75_Think.yaml"

run_eval_original "SocraticLM" "SocraticLM_original.yaml"
run_eval_original "TutorRL" "TutorRL_original.yaml"
run_eval_original "TutorRL-think" "TutorRL-think_original.yaml"
run_eval_original "Qwen2.5-7B" "Qwen2.5-7B-Instruct_original.yaml"
run_eval_original "ConstructivistRL" "ConstructivistRL-GDPO-7b_Soft0.75_Think_original.yaml"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "======================================"
echo "🎉 All Eval Finished : $(date)"
echo "⏱️ Total Time : $((ELAPSED/3600))h $(((ELAPSED%3600)/60))m $((ELAPSED%60))s"
echo "======================================"
