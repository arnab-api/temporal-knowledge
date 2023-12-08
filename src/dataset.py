import copy
import os
from dataclasses import dataclass, replace
from typing import Optional

import src.utils.dataset_utils as dset_utils

import names
import numpy as np
from dataclasses_json import DataClassJsonMixin
from torch.utils.data import Dataset

DEFAULT_PATH = "data/counterfact"


@dataclass(frozen=False)
class QA_Sample(DataClassJsonMixin):
    query: str
    subject: str
    variable: str
    answer: str


@dataclass()
class VariableBindingFactRecallDataset(DataClassJsonMixin, Dataset):
    few_shot_examples: str
    qa_samples: list[QA_Sample]

    def __init__(
        self,
        few_shot_examples: str,
        qa_samples: list[QA_Sample],
    ) -> None:
        super().__init__()
        self.few_shot_examples = few_shot_examples
        self.qa_samples = qa_samples

    # @property
    def __len__(self) -> int:
        return len(self.qa_samples)

    # @property
    def __getitem__(self, idx):
        query = self.qa_samples[idx].query
        answer = self.qa_samples[idx].answer
        full_query = "\n".join(self.few_shot_examples + [query])
        return (full_query, answer)


def generate_synthetic_dataset(
    relation_subj_obj_mapping: list[tuple[str, str]],
    variable_binding_template=" {} is visiting {}",
    query_template=" {} is in {}.",
    num_options: int = 3,  # number of options to choose from in the query
    num_icl: int = 5,  # number of few-shot examples to contextualize
    batch_size: int = 100,
) -> VariableBindingFactRecallDataset:
    (
        icl_examples,
        _,
        _,
        _,
        used_subjects,
        used_variables,
    ) = dset_utils.get_demonstrations(
        relation_subj_obj_mapping,
        num_options,
        num_icl,
        variable_binding_template=variable_binding_template,
        query_template=query_template,
    )

    subj_placeholder_idx = query_template.index("{}")
    obj_placeholder_idx = (
        subj_placeholder_idx
        + 1
        + query_template[query_template.index("{}") + 1 :].index("{}")
    )
    last_query = query_template[:obj_placeholder_idx].rstrip()

    qa_pairs: list[QA_Sample] = []

    # print(used_variables)
    # print(used_subjects)

    indices = list(range(len(relation_subj_obj_mapping)))
    np.random.shuffle(indices)

    # while len(qa_pairs) < batch_size:
    for idx in indices:
        subj, ans = relation_subj_obj_mapping[idx]
        if subj in used_subjects:
            continue
        # print(
        #     f"{len(qa_pairs)}/{batch_size} ==> {len(used_subjects)} | {len(used_variables)}"
        # )
        (
            cur_query,
            cur_answer,
            cur_subject,
            cur_variable,
            _,
            _,
        ) = dset_utils.get_demonstrations(
            subj_obj_mapping=relation_subj_obj_mapping,
            num_options=num_options,
            num_icl=1,
            variable_binding_template=variable_binding_template,
            query_template=last_query,
            used_variables=copy.copy(used_variables),
            used_subjects=copy.copy(used_subjects),
        )
        qa_pairs.append(
            QA_Sample(
                query=cur_query[0],
                subject=cur_subject[0],
                variable=cur_variable[0],
                answer=cur_answer[0],
            )
        )

        if len(qa_pairs) == batch_size:
            break

    return VariableBindingFactRecallDataset(
        few_shot_examples=icl_examples, qa_samples=qa_pairs
    )
