# A Unified Toolkit for Evaluating GUI Agents

GUIEvalKit is an open-source evaluation toolkit for GUI agents, allowing practitioners to easily assess these agents on various (offline) benchmarks. The main goal is to provide an easy-to-use, open-source toolkit that simplifies the evaluation process for researchers and developers, while ensuring that evaluation results can be easily reproduced.

## Requirements and Installation

This work has been tested in the following environment:
* `python == 3.10.12`
* `torch == 2.8.1+cu128`
* `transformers == 4.57.1`
* `vllm == 0.11.0`

### Installation

Install the required dependencies:

```bash
pip install uv

pushd ./guievalkit/
uv venv ur_venv
source ur_venv/bin/activate
uv pip install -r requirements.txt -i accessible_url  # uv won't read accessible url from pip.conf
```

Make sure you have CUDA 12.8.x installed for GPU acceleration with vLLM.

## Supported Models

| Model                                                   | Model Name                                    | Organization |
|---------------------------------------------------------|-----------------------------------------------|--------------|
| [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL)      | `qwen2.5-vl-3/7/32/72b-instruct`              | Alibaba      |
| [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)          | `qwen3-vl-4/8b-instruct/thinking`             | Alibaba      |
| [GUI-Owl](https://github.com/X-PLUG/MobileAgent)        | `gui-owl-7/32b`                               | Alibaba      |
| [UI-Venus](https://github.com/inclusionAI/UI-Venus)     | `ui-venus-navi-7/72b`                         | Ant Group    |
| [UI-TARS](https://github.com/bytedance/UI-TARS)         | `ui-tars-2/7/72b-sft`, `ui-tars-7/72b-dpo`    | Bytedance    |
| [UI-TARS-1.5](https://github.com/bytedance/UI-TARS)     | `ui-tars-1.5-7b`                              | Bytedance    |
| [MagicGUI](https://github.com/MagicAgent-GUI/MagicGUI)  | `magicgui-cpt/rft`                            | Honor        |
| [AgentCPM-GUI](https://github.com/OpenBMB/AgentCPM-GUI) | `agentcpm-gui-8b`                             | ModelBest    |
| [MiMo-VL](https://github.com/XiaomiMiMo/MiMo-VL)        | `mimo-vl-7b-sft/rl`, `mimo-vl-7b-sft/rl-2508` | Xiaomi       |
| [GLM-V](https://github.com/zai-org/GLM-V)               | `glm-4.1v-9b-thinking`, `glm-4.5v`            | Zhipu AI     |


## Supported Benchmarks

| Dataset                                                                                          | Task Name                  | Task      | Description                        |
|--------------------------------------------------------------------------------------------------|----------------------------|-----------|------------------------------------|
| [AndroidControl](https://github.com/google-research/google-research/tree/master/android_control) | `androidcontrol_low/high`  | Agent     | 1680 episodes, (10814 - 653) steps |
| [CAGUI](https://huggingface.co/datasets/openbmb/CAGUI)                                           | `cagui_agent`              | Agent     | 600 episodes, 4516 steps           |
| [GUI Odyssey](https://github.com/OpenGVLab/GUI-Odyssey)                                          | `gui_odyssey`              | Agent     | 1933 episodes, 29426 steps         |
| [AiTZ](https://github.com/IMNearth/CoAT)                                                         | `aitz`                     | Agent     | 506 episodes, 4724 steps           |

## Data Preparation

Please follow the [instructions](./data/README.md) to download and preprocess the datasets.

## Development

### Configuration

Please update the configuration files or objs with your own information:
- **[dataset_info.json](./config/dataset_info.json)**: Configure dataset paths and settings
- **[guieval/config.py](./guieval/config.py)**: `DATASET` for clear type notation and static checking
- **[model_paths.json](./config/model_paths.json)**: Configure default model paths for supported models

### Model Core Implementation
- **[ur_model.py](./guieval/models/ur_model.py)**: Implement ur model's core methods
- **[__init__.py](./guieval/models/__init__.py)**: Register ur model

## Evaluation

### Quick Start

You can use the provided `run.sh` script as a template, or run directly with Python:

```bash
python3 run.py all \
    --setup.datasets cagui_agent \
    --setup.model.model_name agentcpm-gui-8b \
    --setup.eval_mode offline_rule \
    --setup.vllm_mode online
```

### Command Structure

The evaluation command follows this structure:

```bash
python3 run.py <mode> [--setup.<config_path> <value> ...]
```

**Mode Options:**
- `all`: Perform both inference and evaluation (default)
- `infer`: Only perform inference
- `eval`: Only perform evaluation (currently not implemented)

### Configuration Options

#### Dataset Configuration
- `--setup.datasets (str | list)`: Comma-separated list of datasets to evaluate. Supported datasets: `androidcontrol_low`, `androidcontrol_high`, `cagui_agent`, `gui_odyssey`, `aitz`

#### Model Configuration
- `--setup.model.model_name (str)`: Model name from the supported models list (required)
- `--setup.model.model (str)`: Custom model path (optional, defaults to path in `model_paths.json`)
- `--setup.model.model_alias (str)`: Human-readable model identifier for logs (optional, defaults to `model_name`)
- `--setup.model.max_model_len (int)`: Maximum context length (default: 8192)
- `--setup.model.tensor_parallel_size (int)`: Number of GPUs for tensor parallelism (default: 1)
- `--setup.model.data_parallel_size (int)`: Number of GPUs for data parallelism (default: 1)
- `--setup.model.pipeline_parallel_size (int)`: Number of GPUs for pipeline parallelism (default: 1)
- `--setup.model.max_num_batched_tokens (int)`: Maximum batched tokens per inference (default: 4096)
- `--setup.model.max_num_seqs (int)`: Maximum sequences per inference (default: 32)
- `--setup.model.image_limit (int)`: Maximum images per prompt (default: 3)

#### Evaluation Configuration
- `--setup.eval_mode (str)`: Evaluation mode (default: `offline_rule`)
  - `offline_rule`: Evaluate with model off-policy based on predefined rules
  - `semi_online`: Evaluate on-policy with model's own outputs when task succeeds
  - ...
- `--setup.vllm_mode (str)`: vLLM inference mode (default: `online`)
  - `online`: Use vLLM online serving for concurrent generation
  - `offline`: Use vLLM batched generation
- `--setup.enable_thinking (bool)`: Enable thinking mode for models that support it (default: `true`)
- `--setup.batch_size (int)`: Task Batch size for offline vLLM mode (default: 64)
- `--setup.max_concurrent_tasks (int)`: Maximum concurrent tasks for online vLLM mode (default: 128)

#### Output Configuration
- `--setup.output_dir (str)`: Directory to save evaluation results (default: `./outputs`)
- `--setup.log_dir (str)`: Directory to save logs (default: `./logs/guieval`)

### Example: Using run.sh

You can modify `run.sh` to customize your evaluation:

```bash
datasets=androidcontrol_high,gui_odyssey,cagui_agent
model="ui-tars-1.5-7b"
model_path="None"  # or /path/to/specific_model
model_alias="None"  # or custom alias
mode=all
vllm_mode=online
max_model_len=40960
tp=1
dp=8
pp=1
tokens_batch_size=16384
seq_box=32
image_limit=1
concurrent=32
eval_mode=offline_rule
enable_thinking=false

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
``` 

**Please check [here](./docs/results.md) for the detailed evaluation results.**

<!--

## Development Guide

To add new GUI agents and benchmarks to GUIEvalKit, please refer to the [Development Guide](./docs/development.md).

-->

## Acknowledgement

This repo benefits from [AgentCPM-GUI/eval](https://github.com/OpenBMB/AgentCPM-GUI/tree/main/eval) and [VLMEvalKit](https://github.com/open-compass/VLMEvalKit). Thanks for their wonderful works.