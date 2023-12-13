import copy
import json
import logging
import os
from dataclasses import dataclass, replace
from typing import Literal, Optional

import src.utils.dataset_utils as dset_utils

import numpy as np
from dataclasses_json import DataClassJsonMixin
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


DEFAULT_PATH = "data"


@dataclass(frozen=False)
class Sample(DataClassJsonMixin):
    query: str
    subject: str
    object: str
    placeholders: dict[str, str] = None


def fill_template(template: str, sample: Sample, placeholders: list[str]) -> str:
    placeholder_values = sample.placeholders if isinstance(sample, Sample) else sample
    for placeholder in placeholders:
        template = template.replace(placeholder, placeholder_values[placeholder])
    return template.format(
        sample.subject if isinstance(sample, Sample) else sample["subject"]
    )


# @dataclass()
class TemporalRelation(DataClassJsonMixin, Dataset):
    relation_name: str
    prompt_template: str
    samples: list[Sample]
    properties: Optional[dict]

    few_shot_demonstrations: list[str]
    few_shot_samples: list[Sample]

    def __init__(
        self,
        relation_name: str,
        prompt_template: str,
        samples: list[Sample],
        properties: dict,
    ) -> None:
        super().__init__()
        self.relation_name = relation_name
        self.prompt_template = prompt_template
        self.samples = samples
        self.properties = properties

        self.few_shot_demonstrations = []
        self.few_shot_samples = []

        self.select_icl_examples(properties["num_icl"])

        logger.info(
            f'initialized relation -> "{relation_name}" with {len(self)} samples'
        )

    def select_icl_examples(self, num_icl: int) -> list[Sample]:
        if num_icl == 0:
            return

        self.few_shot_demonstrations = []
        self.few_shot_samples = []

        icl_indices = np.random.choice(len(self.samples), size=num_icl, replace=False)
        for idx in icl_indices:
            self.few_shot_samples.append(self.samples[idx])

        for sample in self.few_shot_samples:
            cur_fact = fill_template(
                self.prompt_template,
                sample,
                list(self.properties["placeholders"].keys()),
            )
            cur_fact += f" {sample.object}"
            self.few_shot_demonstrations.append(cur_fact)

        # remove icl samples from the dataset
        icl_indices = sorted(icl_indices, reverse=True)
        for idx in icl_indices:
            self.samples.pop(idx)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        query = self.samples[idx].query
        object = self.samples[idx].object
        full_query = "\n".join(self.few_shot_demonstrations + [query])
        return (full_query, object)


def load_relation(
    relation_file: str,
    num_icl: int = 5,  # number of few-shot examples to contextualize
    batch_size: int = -1,  # number of samples to load. -1 => load all samples
    prompt_idx: Literal[
        # -1,  # use the zero-shot prompt
        0,  # 0 => ... <var> ... <subj> ...
        1,  # 1 => ... <subj> ... <var> ...
    ] = 0,
    default_path: str = DEFAULT_PATH,
) -> TemporalRelation:
    with open(os.path.join(default_path, relation_file)) as f:
        relation_data = json.load(f)

    properties = relation_data["properties"]
    properties["num_icl"] = num_icl
    prompt_template = relation_data["prompt_templates"][prompt_idx]

    raw_samples = relation_data["samples"]
    np.random.shuffle(raw_samples)

    samples: list[Sample] = []
    for sample in raw_samples:
        placeholders = {
            placeholder: sample[placeholder]
            for placeholder in properties["placeholders"].keys()
        }
        samples.append(
            Sample(
                query=fill_template(
                    prompt_template, sample, list(properties["placeholders"].keys())
                ),
                subject=sample["subject"],
                object=sample["object"],
                placeholders=placeholders,
            )
        )
        batch_size -= 1
        if batch_size == 0:
            break

    return TemporalRelation(
        relation_name=relation_data["name"],
        prompt_template=prompt_template,
        samples=samples,
        properties=properties,
    )


def load_dataset(
    relations: list[str] = [],  # load all relations by default
    num_icl: int = 5,  # number of few-shot examples to contextualize
    batch_size: int = -1,  # number of samples to load per relation. -1 => load all samples
    prompt_idx: Literal[
        # -1,  # use the zero-shot prompt
        0,  # 0 => ... <var> ... <subj> ...
        1,  # 1 => ... <subj> ... <var> ...
    ] = 0,
    default_path: str = DEFAULT_PATH,
) -> list[TemporalRelation]:
    if len(relations) == 0:
        relations = list(os.listdir(default_path))
        relations.remove("raw")
    else:
        relations = [f'{r.lower().replace(" ", "_")}.json' for r in relations]

    dataset: list[TemporalRelation] = []

    for relation in relations:
        dataset.append(
            load_relation(
                relation_file=relation,
                num_icl=num_icl,
                batch_size=batch_size,
                prompt_idx=prompt_idx,
                default_path=default_path,
            )
        )

    return dataset
