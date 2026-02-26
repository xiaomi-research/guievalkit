from pydantic import Field, model_serializer
from typing_extensions import TypedDict

# subsec internal
from guieval.utils import UNIFIED_ACTION, PREDICTION
from guieval.main import StepTaskModel, EvaluateResult


# section main
class _TaskSample(TypedDict):
    response: str
    thought: str
    action: UNIFIED_ACTION
    prediction: PREDICTION
    evaluation: EvaluateResult


class StepTaskSampler(StepTaskModel):
    model_alias: str | None = None
    evaluation: EvaluateResult | None = Field(default_factory=EvaluateResult, frozen=True)

    # sampler attributes
    samples: list[_TaskSample] = Field(default_factory=list)

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

    def append_sample(self,
                      response: str,
                      thought: str,
                      action: UNIFIED_ACTION,
                      prediction: PREDICTION):
        new_sample: _TaskSample = {
            'response': response,
            'thought': thought,
            'action': action,
            'prediction': prediction
        }

        self.prediction = prediction
        self.clear_evaluation_cache()

        new_sample['evaluation'] = self.evaluate()

        self.samples.append(new_sample)

        return

    @model_serializer(mode='plain')
    def _serialize_model(self):
        model = super()._serialize_model()

        dumped_samples = list()
        for sample in self.samples:
            sample['evaluation'] = sample['evaluation'].model_dump()
            dumped_samples.append(sample)

        self.samples.clear()

        model['samples'] = dumped_samples

        return model
