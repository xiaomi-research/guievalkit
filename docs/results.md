## Agent Evaluation Results

| Model                              | Android Control-Low | Android Control-High | GUI-Odyssey | AiTZ | CAGUI-Agent |
|------------------------------------|---------------------|----------------------|-------------|------|-------------|
| UI-TARS-2B-SFT                     |                     |                      |             |      |             |
| Qwen2.5-VL-3B                      |                     |                      |             |      |             |
| Qwen2.5-VL-7B                      |                     |                      |             |      |             |
| UI-TARS-7B-SFT                     |                     |                      |             |      |             |
| UI-TARS-7B-DPO                     |                     |                      |             |      |             |
| UI-TARS-1.5-7B                     |                     |                      |             |      |             |
| MiMo-VL-7B-SFT                     |                     |                      |             |      |             |
| MiMo-VL-7B-RL                      |                     |                      |             |      |             |
| MiMo-VL-7B-SFT-2508                |                     |                      |             |      |             |
| MiMo-VL-7B-SFT-2508 (w/o thinking) |                     |                      |             |      |             |
| MiMo-VL-7B-RL-2508                 |                     |                      |             |      |             |
| MiMo-VL-7B-RL-2508 (w/o thinking)  |                     |                      |             |      |             |
| UI-Venus-Navi-7B                   |                     |                      |             |      |             |
| GUI-Owl-7B                         |                     |                      |             |      |             |
| GUI-Owl-7B (w/o thinking)          |                     |                      |             |      |             |
| AgentCPM-GUI-8B                    |                     |                      |             |      |             |
| GLM-4.1V-Thinking                  |                     |                      |             |      |             |
| GUI-Owl-32B                        |                     |                      |             |      |             |
| GUI-Owl-32B (w/o thinking)         |                     |                      |             |      |             |
| UI-TARS-72B-SFT                    |                     |                      |             |      |             |
| UI-TARS-72B-DPO                    |                     |                      |             |      |             |
| UI-Venus-Navi-72B                  |                     |                      |             |      |             |
| GLM-4.5v                           |                     |                      |             |      |             |

Note that:
1. We find `qwen2.5-vl-32/72b-instruct` exhibits significant hallucinations in GUI agent tasks. Therefore, we exclude them from the evaluation.
2. We discard `open_app(app_name='')` action when applying UI-TARS models.
3. We discard `open_app` and `answer` actions when applying GLM-V models.
4. We discard `Launch(app='')` action when applying UI-Venus models.
5. We remove the `OPEN_APP` step when evaluating the AndroidControl benchmark.
