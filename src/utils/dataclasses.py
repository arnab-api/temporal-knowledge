from dataclasses import dataclass
from typing import Any, Optional, Union

from src.dataset import Sample, TemporalRelation
from src.utils.typing import Layer

from dataclasses_json import DataClassJsonMixin


@dataclass(frozen=True)
class PredictedToken(DataClassJsonMixin):
    """A predicted token and its probability."""

    token: str
    prob: float

    def __str__(self) -> str:
        return f'"{self.token}" (p={self.prob:.3f})'


@dataclass(frozen=True)
class SampleResult(DataClassJsonMixin):
    query: str
    answer: str
    prediction: list[PredictedToken]


@dataclass(frozen=True)
class LayerResult(DataClassJsonMixin):
    samples: list[SampleResult]
    score: float


@dataclass(frozen=True)
class TrialResult(DataClassJsonMixin):
    few_shot_demonstration: str
    faithfulness: dict[Layer, LayerResult]
    # efficacy: dict[Layer, float] # TODO: may add this later


@dataclass(frozen=True)
class ExperimentResults(DataClassJsonMixin):
    experiment_specific_args: dict[str, Any]
    trial_results: list[TrialResult]
