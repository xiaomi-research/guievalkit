# A Unified Toolkit for Evaluating GUI Agents

GUIEvalKit is an open-source evaluation toolkit for GUI agents, allowing practitioners to easily assess these agents on various (offline) benchmarks. The main goal is to provide an easy-to-use, open-source toolkit that simplifies the evaluation process for researchers and developers, while ensuring that evaluation results can be easily reproduced.

## Requirements and Installation

This work has been tested in the following environment:
* `python == 3.10.12`
* `torch == 2.7.1+cu126`
* `transformers == 4.56.0.dev0`
* `vllm == 0.10.1`
* `flashinfer-python`
* `qwen-agent`
* `qwen_vl_utils`
* `python-Levenshtein`

## Supported Models

| Model        | Model Name                                                                     | Organization |
|--------------|--------------------------------------------------------------------------------|--------------|
| Qwen2.5-VL   | `qwen2.5-vl-3/7b-instruct`                                                     | Alibaba      |
| GUI-Owl      | `gui-owl-7/32b`                                                                | Alibaba      |
| UI-Venus     | `ui-venus-navi-7b`, `ui-venus-navi-72b`                                        | Ant Group    |
| UI-TARS      | `ui-tars-2/7/72b-sft`, `ui-tars-7/72b-dpo`                                     | Bytedance    |
| UI-TARS-1.5  | `ui-tars-1.5-7b`                                                               | Bytedance    |
| AgentCPM-GUI | `agentcpm-gui-8b`                                                              | ModelBest    |
| MiMo-VL      | `mimo-vl-7b-sft`, `mimo-vl-7b-sft-2508`, `mimo-vl-7b-rl`, `mimo-vl-7b-rl-2508` | Xiaomi       |
| GLM-V        | `glm-4.1v-9b-thinking`, `glm-4.5v`                                             | Zhipu AI     |

1. We find that `qwen2.5-vl-32b-instruct` and `qwen2.5-vl-72b-instruct` exhibit significant hallucinations in GUI agent tasks. Therefore, we exclude them from the evaluation.
2. We discard `"open_app(app_name=\'\')\n"` action when applying UI-TARS models.
3. We discard `open_app` and `answer` actions when applying `glm-4.1v-9b-thinking` and `glm-4.5v`.
4. We discard `"Launch(app='')\n"` action when applying UI-Venus models.

## Supported Benchmarks


| Dataset        | Task Name                                   | Task      | Description                         |
|----------------|---------------------------------------------|-----------|-------------------------------------|
| AndroidControl | `androidcontrol_low`, `androidcontrol_high` | Agent     | 1680 episides, (10814 - 653) steps  |
| CAGUI          | `cagui_agent`                               | Agent     | 600 episodes, 4516 steps            |
| GUI Odyssey    | `gui_odyssey`                               | Agent     | 1933 episodes, 29426 steps          |
| AiTZ           | `aitz`                                      | Agent     | 506 episodes, 4724 steps            |

1. We remove the `OPEN_APP` step when evaluating the AndroidControl benchmark. 



## Acknowledgement

This repo benefits from [AgentCPM-GUI/eval](https://github.com/OpenBMB/AgentCPM-GUI/tree/main/eval) and [VLMEvalKit](https://github.com/open-compass/VLMEvalKit). Thanks for their wonderful works.