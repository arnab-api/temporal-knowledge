import copy
import json
import logging
import os
import re
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
    subject: str
    object: str
    placeholders: dict[str, str] = None

    def __str__(self):
        return f"{self.subject} [{list(self.placeholders.values())}] => {self.object}"


def get_placeholder_keys(template: str) -> list[str]:
    ph_start = [ph.start() for ph in re.finditer(r"<", template)]
    ph_end = [ph.end() for ph in re.finditer(r">", template)]

    placeholders = [template[st:nd] for st, nd in zip(ph_start, ph_end)]
    return placeholders


def fill_template(template: str, sample: str | Sample) -> str:
    if isinstance(sample, str):
        return template.format(sample)
    placeholder_values = sample.placeholders if isinstance(sample, Sample) else sample
    placeholders = get_placeholder_keys(template)
    for placeholder in placeholders:
        template = template.replace(placeholder, placeholder_values[placeholder])
    return template.format(
        sample.subject if isinstance(sample, Sample) else sample["subject"]
    )


@dataclass()
class TemporalRelation(DataClassJsonMixin, Dataset):
    relation_name: str
    prompt_template: str
    prompt_template_zs: Optional[str]
    samples: list[Sample]
    properties: Optional[dict]

    few_shot_demonstrations: list[str]
    few_shot_samples: list[Sample]
    _range: list[str] | None

    def __init__(
        self,
        relation_name: str,
        prompt_template: str,
        samples: list[Sample],
        properties: dict,
        prompt_template_zs: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.relation_name = relation_name
        self.prompt_template = prompt_template
        self.samples = samples
        self.properties = properties
        self._range = None
        self._range = self.range
        self.prompt_template_zs = prompt_template_zs

        self.few_shot_demonstrations = []
        self.few_shot_samples = []

        self.select_icl_examples(properties["num_icl"])

        logger.info(
            f'initialized relation -> "{relation_name}" with {len(self)} samples'
        )

    def select_icl_examples(self, num_icl: int) -> list[Sample]:
        """
        Selects num_icl samples from the dataset to contextualize the relation.
        Considerations:
            - Don't select samples with the same subject
            - Try to balance the number of samples per object
        """
        if num_icl == 0:
            return

        self.few_shot_demonstrations = []
        self.few_shot_samples = []
        icl_indices = []

        iter_indices = np.random.permutation(len(self.samples))
        subj_taken = set()
        obj_limit = {obj: num_icl // len(self.range) for obj in self.range}
        mod = num_icl % len(self.range)

        for idx in iter_indices:
            sample = self.samples[idx]

            # print(sample, obj_limit, mod)

            if sample.subject in subj_taken:
                continue
            if sample.object not in obj_limit:
                if len(obj_limit) != 0:
                    continue

            self.few_shot_samples.append(sample)
            icl_indices.append(idx)

            subj_taken.add(sample.subject)
            if sample.object in obj_limit:
                obj_limit[sample.object] -= 1
                if obj_limit[sample.object] == 0:
                    obj_limit.pop(sample.object)
            else:
                mod -= 1
                if mod == 0:
                    break

            if len(self.few_shot_samples) == num_icl:
                break

        np.random.shuffle(self.few_shot_samples)
        for sample in self.few_shot_samples:
            cur_fact = fill_template(
                self.prompt_template,
                sample,
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
        query = fill_template(template=self.prompt_template, sample=self.samples[idx])
        object = self.samples[idx].object
        full_query = "\n".join(self.few_shot_demonstrations + [query])
        return (full_query, object)

    @property
    def range(self):
        if self._range is None:
            self._range = [sample.object for sample in self.samples]
            self._range = list(set(self._range))
        if len(self._range) == 1:
            logger.warning(
                f"*** WARNING: range has only one element {self._range} ==> Trivial task! ***"
            )
        return self._range

    @property
    def range_stats(self):
        counts = {}
        for sample in self.samples:
            if sample.object not in counts:
                counts[sample.object] = 0
            counts[sample.object] += 1
        return counts


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
    prompt_template_zs = relation_data["prompt_templates_zs"][prompt_idx]

    raw_samples = relation_data["samples"]
    np.random.shuffle(raw_samples)

    samples: list[Sample] = []
    _range = []
    for sample in raw_samples:
        placeholders = {
            placeholder: sample[placeholder]
            for placeholder in properties["placeholders"].keys()
        }
        samples.append(
            Sample(
                subject=sample["subject"],
                object=sample["object"],
                placeholders=placeholders,
            )
        )
        _range.append(sample["object"])

    _range = list(set(_range))
    if batch_size != -1:
        if len(_range) < batch_size:
            samples = balance_samples(samples, num_per_obj=batch_size // len(_range))
        else:
            samples = balance_samples(samples)

    return TemporalRelation(
        relation_name=relation_data["name"],
        prompt_template=prompt_template,
        samples=samples,
        properties=properties,
        prompt_template_zs=prompt_template_zs,
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


def balance_samples(samples: list[Sample], num_per_obj: int | None = None):
    """
    Balance the number of samples per object in the relation
    """
    samples_per_obj = {}
    for sample in samples:
        if sample.object not in samples_per_obj:
            samples_per_obj[sample.object] = []
        samples_per_obj[sample.object].append(sample)

    min_samples = min([len(samples) for samples in samples_per_obj.values()])
    min_samples = (
        min(min_samples, num_per_obj) if num_per_obj is not None else min_samples
    )
    new_samples = []
    for samples in samples_per_obj.values():
        new_samples.extend(samples[:min_samples])

    np.random.shuffle(new_samples)
    return new_samples
