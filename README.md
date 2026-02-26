# GUIEvalKit: A Unified Toolkit for Evaluating GUI Agents

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg">
  <img src="https://img.shields.io/badge/vLLM-Supported-orange.svg">
  <img src="https://img.shields.io/github/license/google-research/google-research.svg">
  <img src="https://img.shields.io/github/stars/gaopengzhi/guievalkit?style=social">
</p>

**GUIEvalKit** is an open-source, high-performance evaluation toolkit designed for next-generation GUI agents. It provides a unified interface to assess multimodal models (LMMs) across various offline benchmarks, ensuring reproducibility and efficiency.

---

## 🌟 Key Features

*   🚀 **Unified API**: Support for 10+ SOTA GUI models (UI-TARS, Qwen2.5-VL, GLM-4.5v, etc.) with a single interface.
*   🧠 **Reasoning Support**: Native support for "Thinking" models (e.g., Qwen3-VL-Thinking, GLM-4.1V-Thinking).
*   ⚡ **High Performance**: Optimized with **vLLM** for both online serving and offline batch inference.
*   📊 **Comprehensive Benchmarks**: Ready-to-use evaluation on AndroidControl, GUI Odyssey, AiTZ, and more.
*   🛠️ **Easy Extension**: Add new models or datasets by implementing just a few methods.

---

## 🏆 Leaderboard (Preview)

Below is a summary of evaluation results (Step Success Rate %). For full results, see [results.md](./docs/results.md).

| Model | Android Control (High) | GUI-Odyssey | AiTZ | CAGUI |
| :--- | :---: | :---: | :---: | :---: |
| **UI-TARS-72B-SFT** | **79.37** | 72.27 | 69.83 | 74.53 |
| **AgentCPM-GUI-8B** | 67.93 | **74.84** | **76.08** | **91.32** |
| **UI-Venus-Navi-72B**| 73.53 | 72.10 | 65.20 | 69.60 |
| **Qwen2.5-VL-7B** | 61.40 | 47.92 | 64.73 | 58.48 |
| **GLM-4.5v** | 59.15 | 48.90 | 48.52 | 69.26 |

---

## 🛠️ Installation

We recommend using `uv` for lightning-fast dependency management.

```bash
pip install uv
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt
```

---

## 🚀 Quick Start

Evaluate a model on a specific benchmark with one command:

```bash
python3 run.py all \
    --setup.datasets cagui_agent \
    --setup.model.model_name agentcpm-gui-8b \
    --setup.vllm_mode online
```

---

## 📂 Supported Models & Benchmarks

<details>
<summary><b>Click to expand supported models</b></summary>

| Organization | Models |
| :--- | :--- |
| **Alibaba** | Qwen2.5-VL, Qwen3-VL, GUI-Owl |
| **Bytedance** | UI-TARS (SFT/DPO), UI-TARS-1.5 |
| **Zhipu AI** | GLM-4.1V-Thinking, GLM-4.5v |
| **Ant Group** | UI-Venus-Navi |
| **ModelBest** | AgentCPM-GUI |
| **Others** | MiMo-VL, MagicGUI |
</details>

<details>
<summary><b>Click to expand supported benchmarks</b></summary>

*   **AndroidControl**: Comprehensive Android task execution.
*   **GUI Odyssey**: Cross-platform GUI navigation tasks.
*   **AiTZ**: Action-in-the-Wild datasets.
*   **CAGUI**: Large-scale GUI agent benchmark.
</details>

---

## 🏗️ Development & Contribution

Want to add your own model? It's as simple as inheriting from `ABCModel`. Check our [Development Guide](./docs/development.md) for details.

We welcome contributions! Please feel free to submit PRs or open issues.

---

## 📜 Acknowledgement

This project is built upon the great works of [AgentCPM-GUI/eval](https://github.com/OpenBMB/AgentCPM-GUI/tree/main/eval) and [VLMEvalKit](https://github.com/open-compass/VLMEvalKit).

---

## 📑 Citation

If you find this toolkit useful for your research, please consider citing:

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
