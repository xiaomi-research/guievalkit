# Development Guide

Welcome to GUIEvalKit! This guide will help you add new models and benchmarks to the toolkit.

## 1. Adding a New Model

To add a new GUI agent model, follow these steps:

### Step 1: Create a Model Class
Create a new file in `guieval/models/` (e.g., `guieval/models/my_new_model.py`) and inherit from `ABCModel`.

```python
from guieval.models.abcmodel import ABCModel, MODEL_ACTION, MINICPM_ACTION, RawInput, ParsedResponse
from guieval.models.utils import ModelPatterns

class MyNewModel(ABCModel):
    NAMES = ("my-model-name",)
    MODEL_PATTERNS = ModelPatterns(
        answer_pattern=r"Action: (.*)",
        thinking_pattern=r"Thought: (.*)"
    )

    def parse_response(self, resp: str) -> ParsedResponse:
        # Parse the raw string from model to structured thought/action
        ...

    def model_2_minicpm(self, output_text: str, width: int, height: int) -> MINICPM_ACTION:
        # Convert model output to unified MiniCPM action format (normalized coordinates)
        ...

    def aitw_2_model_action(self, step_task, height, width) -> MODEL_ACTION:
        # Map ground truth AITW actions to your model's action space
        ...

    def prepare_task_input(self, step_task, **kwargs) -> RawInput:
        # Formulate the prompt (messages) for your model
        ...
```

### Step 2: Register the Model
Add your model to `guieval/models/__init__.py` or the registry.

### Step 3: Configure Model Path
Add the default path or identifier in `config/model_paths.json`.

---

## 2. Adding a New Benchmark

1. **Preprocessing**: Create a script in `data/` to convert the raw dataset to the unified format used by GUIEvalKit.
2. **Dataset Info**: Add the dataset details to `config/dataset_info.json`.
3. **Task Definition**: If the dataset introduces new action types, update `guieval/utils/action_space.py`.

## 3. Running Tests

Before submitting a PR, ensure your model works with the quickstart command:
```bash
python3 run.py all --setup.model.model_name my-model-name --setup.datasets cagui_agent
```
