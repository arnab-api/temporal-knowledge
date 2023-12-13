import copy
import logging
from typing import Any, Callable, Literal, Optional, Union

import src.utils.tokenizer_utils as tokenizer_utils
from src.dataset import TemporalRelation
from src.models import ModelandTokenizer

import baukit
import torch

logger = logging.getLogger(__name__)


def find_token_range(
    string: str,
    substring: str,
    tokenizer=None,
    occurrence: int = 0,
    offset_mapping=None,
    **kwargs: Any,
) -> tuple[int, int]:
    """Find index range of tokenized string containing tokens for substring.

    The kwargs are forwarded to the tokenizer.

    A simple example:

        string = 'The batman is the night.'
        substring = 'batman'
        tokenizer = ...

        # Example tokenization: ['the', 'bat', '##man', 'is', 'the', 'night']
        assert find_token_range(string, substring, tokenizer) == (1, 3)

    Args:
        string: The string.
        substring: The substring to find token range for.
        tokenizer: The tokenizer. If not set, offset_mapping must be.
        occurrence: The occurence of the substring to look for.
            Zero indexed. Defaults to 0, the first occurrence.
        offset_mapping: Precomputed offset mapping. If not set, tokenizer will be run.

    Raises:
        ValueError: If substring is not actually in string or if banned
            kwargs are specified.

    Returns:
        Tuple[int, int]: The start (inclusive) and end (exclusive) token idx.
    """
    if tokenizer is None and offset_mapping is None:
        raise ValueError("must set either tokenizer= or offset_mapping=")
    if "return_offsets_mapping" in kwargs:
        raise ValueError("cannot set return_offsets_mapping")
    if substring not in string:
        raise ValueError(f'"{substring}" not found in "{string}"')
    if occurrence < 0:
        # If occurrence is negative, count from the right.
        char_start = string.rindex(substring)
        for _ in range(-1 - occurrence):
            try:
                char_start = string.rindex(substring, 0, char_start)
            except ValueError as error:
                raise ValueError(
                    f"could not find {-occurrence} occurrences "
                    f'of "{substring} in "{string}"'
                ) from error
    else:
        char_start = string.index(substring)
        for _ in range(occurrence):
            try:
                char_start = string.index(substring, char_start + 1)
            except ValueError as error:
                raise ValueError(
                    f"could not find {occurrence + 1} occurrences "
                    f'of "{substring} in "{string}"'
                ) from error
    char_end = char_start + len(substring)

    if offset_mapping is None:
        assert tokenizer is not None
        tokens = tokenizer(string, return_offsets_mapping=True, **kwargs)
        offset_mapping = tokens.offset_mapping

    token_start, token_end = None, None
    for index, (token_char_start, token_char_end) in enumerate(offset_mapping):
        # Skip special tokens # ! Is this the proper way to do this?
        if token_char_start == token_char_end:
            continue
        if token_start is None:
            if token_char_start <= char_start and token_char_end >= char_start:
                token_start = index
        if token_end is None:
            if token_char_start <= char_end and token_char_end >= char_end:
                token_end = index
                break

    assert token_start is not None
    assert token_end is not None
    assert token_start <= token_end
    return (token_start, token_end + 1)


from src.utils.dataclasses import PredictedToken


@torch.inference_mode()
def predict_next_token(
    mt: ModelandTokenizer,
    prompt: Union[str, list[str]],
    k: int = 5,
    batch_size: int = 8,
    token_of_interest: Optional[Union[Union[str, int], list[Union[str, int]]]] = None,
) -> Union[
    list[list[PredictedToken]],
    tuple[list[list[PredictedToken]], list[tuple[int, PredictedToken]]],
]:
    # ! Do not use batch decoding for LLaMA-2 models. Not working perfectly.
    """Compute the next token."""
    if isinstance(prompt, str):
        prompt = [prompt]
        if token_of_interest is not None:
            token_of_interest = [token_of_interest]

    if token_of_interest is not None:
        assert len(token_of_interest) == len(prompt)
        track_interesting_tokens = []

    with tokenizer_utils.set_padding_side(mt.tokenizer, padding_side="left"):
        inputs = mt.tokenizer(prompt, return_tensors="pt", padding="longest").to(
            mt.model.device
        )

    with torch.inference_mode():
        predictions = []
        for i in range(0, len(inputs.input_ids), batch_size):
            batch_outputs = mt.model(
                input_ids=inputs.input_ids[i : i + batch_size],
                attention_mask=inputs.attention_mask[i : i + batch_size],
            )

            next_token_probs = batch_outputs.logits[:, -1].float().softmax(dim=-1)
            next_token_topk = next_token_probs.topk(dim=-1, k=k)

            for token_ids, token_probs in zip(
                next_token_topk.indices, next_token_topk.values
            ):
                predictions.append(
                    [
                        PredictedToken(
                            token=mt.tokenizer.decode(token_id),
                            token_id=token_id.item(),
                            prob=prob.item(),
                        )
                        for token_id, prob in zip(token_ids, token_probs)
                    ]
                )
            if token_of_interest is not None:
                for j in range(i, i + batch_outputs.logits.shape[0]):
                    tok_id = (
                        mt.tokenizer(token_of_interest[j]).input_ids[1]
                        if type(token_of_interest[j]) == str
                        else token_of_interest[j]
                    )
                    probs = next_token_probs[j - i]
                    order = probs.topk(dim=-1, k=probs.shape[-1]).indices.squeeze()
                    prob_tok = probs[tok_id]
                    rank = int(torch.where(order == tok_id)[0].item() + 1)
                    track_interesting_tokens.append(
                        (
                            rank,
                            PredictedToken(
                                token=mt.tokenizer.decode(tok_id),
                                token_id=tok_id.item()
                                if isinstance(tok_id, torch.Tensor)
                                else tok_id,
                                prob=prob_tok.item(),
                            ),
                        )
                    )
    if token_of_interest is not None:
        return predictions, track_interesting_tokens
    return predictions


def any_is_nontrivial_prefix(predictions: list[str], target: str) -> bool:
    """Return true if any prediction is (case insensitive) prefix of the target."""
    return any(is_nontrivial_prefix(p, target) for p in predictions)


def is_nontrivial_prefix(prediction: str, target: str) -> bool:
    """Return true if prediction is (case insensitive) prefix of the target."""
    target = target.lower().strip()
    prediction = prediction.lower().strip()
    return len(prediction) > 0 and target.startswith(prediction)


def get_tick_marker(value: bool) -> str:
    """Returns a tick or cross marker depending on the value."""
    return "✓" if value else "✗"


def format_whitespace(s: str) -> str:
    """Format whitespace in a string for printing."""
    return s.replace("\n", "\\n").replace("\t", "\\t")


def make_icl_prompt(
    icl_examples: list, prompt_template: str, bos_token: str = "", subject: str = {}
):
    assert prompt_template.count("{}") == 1
    prompt = (
        bos_token
        + " "
        + "\n".join(
            [
                prompt_template.format(example[0]) + f" {example[1]}"
                for example in icl_examples
            ]
        )
    )
    prompt += "\n" + prompt_template.format(subject)
    return prompt


def filter_samples_by_var(relation: TemporalRelation, var: str) -> TemporalRelation:
    filtered_samples = [
        sample
        for sample in relation.samples + relation.few_shot_samples
        if sample.placeholders["<var>"] == var
    ]

    logger.info(
        f"filtered {len(filtered_samples)} with var={var}, from {relation.relation_name}"
    )

    filtered_relation = copy.deepcopy(relation)
    filtered_relation.samples = filtered_samples
    filtered_relation.select_icl_examples(relation.properties["num_icl"])
    return filtered_relation


@torch.inference_mode()
def filter_samples_by_model_knowledge(
    mt: ModelandTokenizer, relation: TemporalRelation
) -> TemporalRelation:
    """Filter samples by model knowledge."""
    logger.debug(f'"{relation.relation_name}" | filtering with {mt.model_name}')

    filtered_samples = []
    for i in range(len(relation.samples)):
        question, answer = relation[i]
        predictions = predict_next_token(mt, question, k=5)[0]
        top_pred = predictions[0]
        is_known = is_nontrivial_prefix(prediction=top_pred.token, target=answer)
        sample = relation.samples[i]
        if is_known:
            filtered_samples.append(sample)

        logger.debug(
            f"{sample.subject=}[{sample.placeholders['<var>']}] -> {answer=} | predicted = '{top_pred.token}'({top_pred.prob:.3f}) ==> ({get_tick_marker(is_known)})"
        )

    logger.info(
        f'filtered relation "{relation.relation_name}" to {len(filtered_samples)} samples (with {relation.properties["num_icl"]}-shots)'
    )

    relation.samples = filtered_samples
    return relation


def untuple(x):
    if isinstance(x, tuple):
        return x[0]
    return x


def patch_output(
    patch_layer: str, patch_idx: int, patching_vector: torch.Tensor
) -> Callable:
    def edit_output(layer, output):
        if layer != patch_layer:
            return output
        untuple(output)[:, patch_idx] = patching_vector
        return output

    return edit_output


@torch.inference_mode()
def get_h(
    mt: ModelandTokenizer,
    prompt: str,
    subject: str,
    layers: list[str],
    mode: Literal["input", "output"] = "output",
) -> dict[str, torch.Tensor]:
    tokenized = mt.tokenizer(
        prompt, return_offsets_mapping=True, return_tensors="pt"
    ).to(mt.model.device)
    offset_mapping = tokenized.pop("offset_mapping")[0]
    if "token_type_ids" in tokenized:
        tokenized.pop("token_type_ids")

    subject_start, subject_end = find_token_range(
        prompt, subject, tokenizer=mt.tokenizer, offset_mapping=offset_mapping
    )

    subj_last_idx = subject_end - 1
    logger.debug(
        f"edit_index={subj_last_idx} | edit_token={mt.tokenizer.decode(tokenized['input_ids'][0][subj_last_idx])}"
    )

    retain_input = mode == "input"
    with baukit.TraceDict(
        module=mt.model, layers=layers, retain_input=retain_input
    ) as traces:
        mt.model(**tokenized)

    h = {
        layer: untuple(traces[layer].output)[:, subject_end - 1].squeeze()
        if mode == "output"
        else untuple(traces[layer].input)[:, subject_end - 1].squeeze()
        for layer in layers
    }
    return h
