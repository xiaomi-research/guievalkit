#!/bin/bash
set -e # Exit on any error

# Configuration with defaults
datasets=${1:-"androidcontrol_high,gui_odyssey,cagui_agent"}
model=${2:-"ui-tars-1.5-7b"}
model_path=${3:-"None"}
model_alias=${4:-"None"}
mode=${5:-"all"}
vllm_mode=${6:-"online"}
max_model_len=${7:-40960}
tp=${8:-1}
dp=${9:-8}
pp=${10:-1}
tokens_batch_size=${11:-16384}
seq_box=${12:-32}
image_limit=${13:-1}
concurrent=${14:-32}
eval_mode=${15:-"offline_rule"}
enable_thinking=${16:-"false"}

echo "🚀 Starting GUIEvalKit with the following configuration:"
echo "   Datasets: $datasets"
echo "   Model:    $model"
echo "   Mode:     $mode ($vllm_mode)"
echo "   TP/DP/PP: $tp/$dp/$pp"
echo "--------------------------------------------------------"

python3 run.py "${mode}" \
    --setup.datasets "${datasets}" \
    --setup.model.model_name "${model}" \
    --setup.model.model_alias "${model_alias}" \
    --setup.model.model "${model_path}" \
    --setup.model.max_model_len "${max_model_len}" \
    --setup.model.tensor_parallel_size "${tp}" \
    --setup.model.data_parallel_size "${dp}" \
    --setup.model.pipeline_parallel_size "${pp}" \
    --setup.model.max_num_batched_tokens "${tokens_batch_size}" \
    --setup.model.max_num_seqs "${seq_box}" \
    --setup.model.image_limit "${image_limit}" \
    --setup.eval_mode "${eval_mode}" \
    --setup.vllm_mode "${vllm_mode}" \
    --setup.max_concurrent_tasks "${concurrent}" \
    --setup.enable_thinking "${enable_thinking}"
