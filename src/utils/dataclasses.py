from dataclasses import dataclass
from typing import Any, Optional, Union

from src.dataset import Sample, TemporalRelation

from dataclasses_json import DataClassJsonMixin


@dataclass(frozen=True)
class PredictedToken(DataClassJsonMixin):
    """A predicted token and its probability."""

    token: str
    token_id: int
    prob: float

    def __str__(self) -> str:
        return f'"{self.token}" (p={self.prob:.3f})'


@dataclass(frozen=True)
class SampleResult(DataClassJsonMixin):
    query: str
    answer: str
    prediction: list[PredictedToken]


@dataclass(frozen=True)
class TrialResult(DataClassJsonMixin):
    few_shot_demonstration: str
    samples: list[SampleResult]
    recall: list[float]


@dataclass(frozen=True)
class PatchingResults_for_one_pair(DataClassJsonMixin):
    source_QA: Sample
    edit_QA: Sample
    edit_index: int
    edit_token: str
    predictions_after_patching: dict[int, list[PredictedToken]]
    rank_edit_ans_after_patching: dict[int, int]


@dataclass(frozen=True)
class LayerPatchingEfficacy(DataClassJsonMixin):
    layer_idx: int
    recall: list[float]
    reciprocal_rank: float


@dataclass(frozen=True)
class PatchingTrialResult(DataClassJsonMixin):
    few_shot_demonstration: list[Sample]
    layer_patching_effecacy: list[LayerPatchingEfficacy]
    patching_results: list[PatchingResults_for_one_pair]


@dataclass(frozen=True)
class ExperimentResults(DataClassJsonMixin):
    experiment_specific_args: dict[str, Any]
    trial_results: Union[list[TrialResult], list[PatchingTrialResult]]
