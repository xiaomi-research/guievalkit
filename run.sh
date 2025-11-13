datasets=androidcontrol_high,gui_odyssey,cagui_agent
model="ui-tars-1.5-7b"  # model_core
model_path="None"  # /path/to/specific_model
model_alias="None" # "ui-tars-1.5-7b-specific-ckpt"
mode=all
vllm_mode=online  # online offline
max_model_len=40960
tp=1
dp=8
pp=1
tokens_batch_size=16384
seq_box=32
image_limit=1
concurrent=32

# control
eval_mode=offline_rule  # offline_rule semi_online
enable_thinking=false  # true, false

python3 run.py ${mode} \
    --setup.datasets ${datasets} \
    --setup.model.model_name ${model} \
    --setup.model.model_alias ${model_alias} \
    --setup.model.model ${model_path} \
    --setup.model.max_model_len ${max_model_len} \
    --setup.model.tensor_parallel_size ${tp} \
    --setup.model.data_parallel_size ${dp} \
    --setup.model.pipeline_parallel_size ${pp} \
    --setup.model.max_num_batched_tokens ${tokens_batch_size} \
    --setup.model.max_num_seqs ${seq_box} \
    --setup.model.image_limit ${image_limit} \
    --setup.eval_mode ${eval_mode} \
    --setup.vllm_mode ${vllm_mode} \
    --setup.max_concurrent_tasks ${concurrent} \
    --setup.enable_thinking ${enable_thinking}
