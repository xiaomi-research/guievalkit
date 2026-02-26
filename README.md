# 🎮 GUIEvalKit: The Ultimate Toolkit for GUI Agent Evaluation

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?logo=pytorch&logoColor=white">
  <img src="https://img.shields.io/badge/vLLM-Accelerated-orange.svg?logo=nvidia&logoColor=white">
  <img src="https://img.shields.io/github/license/gaopengzhi/guievalkit?color=green">
  <img src="https://img.shields.io/github/stars/gaopengzhi/guievalkit?style=social">
</p>

---

**GUIEvalKit** is an open-source, high-performance evaluation toolkit designed for the next generation of GUI-driven AI agents. It provides a unified, efficient interface to benchmark Vision-Language Models (VLMs) across diverse digital environments (Android, Web, Desktop).

### Why GUIEvalKit?
*   🎯 **Unified Interface**: One command to evaluate 10+ SOTA GUI models (UI-TARS, Qwen2.5-VL, GLM-4.5v).
*   ⚡ **Inference Speed**: Native **vLLM** integration for lightning-fast batch inference and online serving.
*   🧠 **Advanced Reasoning**: Full support for "Chain-of-Thought" and "Thinking" models.
*   🎨 **Visual Analytics**: Built-in tools to visualize agent actions and failure cases.
*   📊 **Ready-to-Use Benchmarks**: Out-of-the-box support for AndroidControl, GUI-Odyssey, AiTZ, and CAGUI.

---

## 🏆 Leaderboard

| Model | Android Control (High) | GUI-Odyssey | AiTZ | CAGUI |
| :--- | :---: | :---: | :---: | :---: |
| **UI-TARS-72B-SFT** | **79.37** | 72.27 | 69.83 | 74.53 |
| **AgentCPM-GUI-8B** | 67.93 | **74.84** | **76.08** | **91.32** |
| **UI-Venus-Navi-72B**| 73.53 | 72.10 | 65.20 | 69.60 |
| **Qwen2.5-VL-7B** | 61.40 | 47.92 | 64.73 | 58.48 |
| **GLM-4.5v** | 59.15 | 48.90 | 48.52 | 69.26 |

*See [Full Results](./docs/results.md) for detailed metrics and model configurations.*

---

## 🚀 Quick Start

### 1. Installation
We recommend using `uv` for 10x faster installation.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 2. Run Evaluation
Evaluate any model on any benchmark with a single command:

```bash
python3 run.py all \
    --setup.datasets cagui_agent \
    --setup.model.model_name agentcpm-gui-8b \
    --setup.vllm_mode online
```

---

## 🎨 Visualizing Agent Actions
GUIEvalKit includes a powerful **Profiler** that helps you understand *why* your agent failed. It generates overlays of the model's predicted actions on the original UI screenshots.

*(GIF/Image placeholder: Action visualization demo)*

---

## 📂 Ecosystem Support

<details>
<summary><b>Supported Models (15+)</b></summary>

*   **Alibaba**: Qwen2.5-VL, Qwen3-VL, GUI-Owl
*   **Bytedance**: UI-TARS (SFT/DPO), UI-TARS-1.5
*   **Zhipu AI**: GLM-4.1V-Thinking, GLM-4.5v
*   **Ant Group**: UI-Venus-Navi
*   **ModelBest**: AgentCPM-GUI
*   **DeepSeek**: DeepSeek-VL2
*   **Others**: MiMo-VL, MagicGUI
</details>

<details>
<summary><b>Supported Benchmarks</b></summary>

*   **AndroidControl**: Complex Android task execution.
*   **GUI-Odyssey**: Large-scale cross-platform GUI navigation.
*   **AiTZ (Action-in-the-Wild)**: Real-world smartphone tasks.
*   **CAGUI**: Collaborative Android GUI evaluation.
</details>

---

## 🛠️ Development & Customization

GUIEvalKit is built for extensibility. Adding a new model is as simple as:

1.  Inherit from `ABCModel`.
2.  Implement `prepare_task_input` and `parse_response`.
3.  Register your model in the registry.

Check our [Development Guide](./docs/development.md) for more details.

---

## 🗺️ Roadmap
- [ ] Support for **Desktop GUI** (Windows/Linux) evaluation.
- [ ] Integration with **Gymnasium** for RL-based agent training.
- [ ] **Web GUI** benchmark (WebShop, Mind2Web) integration.
- [ ] Interactive HTML report generation for evaluation results.

---

## 🤝 Contributing & Support
We welcome contributions! Please open an issue or submit a PR. If you like this project, please consider giving it a ⭐!

## 📑 Citation
```bibtex
@misc{guievalkit2025,
  author = {Pengzhi Gao},
  title = {GUIEvalKit: A Unified Toolkit for Evaluating GUI Agents},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/gaopengzhi/guievalkit}}
}
```
