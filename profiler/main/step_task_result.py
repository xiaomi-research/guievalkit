import jinja2
import re

from PIL import Image
from pydantic import Field, model_serializer
from typing import ClassVar
from typing_extensions import TypedDict

# subsec internal
from guieval.main.config import DATASET
from guieval.main import StepTaskModel, EvaluateResult
from guieval.main.step_task import MODE
from profiler.visualize_action import Visualizer
from profiler.utils import RESOURCE_BASE


# section struc
class _StepTaskInfo(TypedDict):
    model_alias: str | None
    mode: MODE
    dataset: DATASET
    episode: str | None
    step: int | str | None
    match: bool | str | None


# section main
class StepTaskResult(StepTaskModel):
    model_alias: str | None = None
    evaluation: EvaluateResult | None = Field(default_factory=EvaluateResult)

    cluster: dict[str, list] = Field(default_factory=dict)
    clustered_decisions: dict[str, dict] = Field(default_factory=dict)
    clustered_distribution: dict[str, int] = Field(default_factory=dict)

    tec: bool | None = None

    hash_template: ClassVar[jinja2.Template] = jinja2.Template((RESOURCE_BASE /
                                                                'visualization_hash_template.j2').read_text())

    def __repr__(self):
        name = type(self).__name__
        evaluation_name = EvaluateResult.__name__
        type_match = (None
                      if self.evaluation is None else
                      self.evaluation.type_match)
        exact_match = (None
                       if self.evaluation is None else
                       self.evaluation.exact_match)
        evaluation = f"{evaluation_name}({type_match=}, {exact_match=}, ...)"
        return self._repr_template.render(name=name, step_task=self, evaluation=evaluation)

    def assign_step_task_result(self, result):
        super().assign_step_task_result(result)
        self.evaluation = EvaluateResult.model_validate(result["evaluation"])

    def evaluate(self):
        return self.evaluation

    @staticmethod
    def hash_decode(hash_code: str) -> _StepTaskInfo:
        model_alias, mode, dataset, episode, step, match = re.search(
            r'<model_alias>(.*)<mode>(.*)<dataset>(.*)<episode>(.*)<step>(.*)<match>(.*)',
            hash_code).groups()

        return {
            'model_alias': model_alias,
            'mode': mode,
            'dataset': dataset,
            'episode': episode,
            'step': step,
            'match': match
        }

    def visualize_prediction(self) -> Image.Image | None:
        assert len(self.image_abspaths) == 1
        screen = Image.open(self.image_abspaths[0])

        return Visualizer.visualize(
            screen,
            self.pred_action,
            self.prediction,
        )

    @model_serializer(mode='plain')
    def _serialize_model(self) -> dict:
        serialized = super()._serialize_model()
        serialized['evaluation'] = self.evaluation.model_dump()

        return serialized
