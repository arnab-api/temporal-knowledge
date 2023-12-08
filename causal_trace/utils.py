import re
from typing import Optional

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


class ModelandTokenizer:
    def __init__(
        self,
        model: Optional[transformers.AutoModel] = None,
        tokenizer: Optional[transformers.AutoTokenizer] = None,
        model_path: Optional[str] = "EleutherAI/gpt-j-6B",
        fp16: bool = True,
    ) -> None:
        assert (
            model is not None or model_path is not None
        ), "Either model or model_name must be provided"
        if model is not None:
            assert tokenizer is not None, "Tokenizer must be provided with the model"
            self.model_name = model.config._name_or_path
        else:
            model, tokenizer = (
                AutoModelForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float16 if fp16 else torch.float32,
                ).to("cuda"),
                AutoTokenizer.from_pretrained(
                    model_path,
                    # padding_side='left'
                ),
            )
            tokenizer.pad_token = tokenizer.eos_token
            model.eval()
            print(
                f"loaded model <{model_path}> | size: {get_model_size(model) :.3f} MB"
            )
            self.model_name = model_path

        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()
        self.parse_config(model.config)

    def parse_config(self, model_config) -> None:
        fields = {
            "n_layer": None,
            "layer_name_format": None,
            "layer_names": None,
            "mlp_module_name_format": None,
            "attn_module_name_format": None,
            "embedder_name": None,
            "final_layer_norm_name": None,
            "lm_head_name": None,
        }
        if (
            "mistral" in model_config._name_or_path.lower()
            or "llama" in model_config._name_or_path.lower()
        ):
            fields["n_layer"] = model_config.num_hidden_layers
            fields["layer_name_format"] = "model.layers.{}"
            fields["mlp_module_name_format"] = "model.layers.{}.mlp"
            fields["attn_module_name_format"] = "model.layers.{}.self_attn"
            fields["embedder_name"] = "model.embed_tokens"
            fields["final_layer_norm_name"] = "model.norm"
            fields["lm_head_name"] = "model.lm_head"

        elif "gpt-j" in model_config._name_or_path.lower():
            fields["n_layer"] = model_config.n_layer
            fields["layer_name_format"] = "transformer.h.{}"
            fields["mlp_module_name_format"] = "transformer.h.{}.mlp"
            fields["attn_module_name_format"] = "transformer.h.{}.attn"
            fields["embedder_name"] = "transformer.wte"
            fields["final_layer_norm_name"] = "transformer.ln_f"
            fields["lm_head_name"] = "transformer.lm_head"

        if fields["layer_name_format"] is not None and fields["n_layer"] is not None:
            fields["layer_names"] = [
                fields["layer_name_format"].format(i) for i in range(fields["n_layer"])
            ]

        for key, value in fields.items():
            if value is None:
                print(f"! Warning: {key} could not be set!")
            setattr(self, key, value)


def make_inputs(tokenizer, prompts, device="cuda"):
    token_lists = [tokenizer.encode(p) for p in prompts]
    maxlen = max(len(t) for t in token_lists)
    if "[PAD]" in tokenizer.all_special_tokens:
        pad_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index("[PAD]")]
    else:
        pad_id = 0
    input_ids = [[pad_id] * (maxlen - len(t)) + t for t in token_lists]
    # position_ids = [[0] * (maxlen - len(t)) + list(range(len(t))) for t in token_lists]
    attention_mask = [[0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists]
    return dict(
        input_ids=torch.tensor(input_ids).to(device),
        #    position_ids=torch.tensor(position_ids).to(device),
        attention_mask=torch.tensor(attention_mask).to(device),
    )


def predict_from_input(model, inp):
    out = model(**inp)["logits"]
    probs = torch.softmax(out[:, -1], dim=1)
    p, preds = torch.max(probs, dim=1)
    return preds, p


def predict_token(mt, prompts, return_p=False):
    inp = make_inputs(mt.tokenizer, prompts)
    preds, p = predict_from_input(mt.model, inp)
    result = [mt.tokenizer.decode(c) for c in preds]
    if return_p:
        result = (result, p)
    return result


def decode_tokens(tokenizer, token_array):
    if hasattr(token_array, "shape") and len(token_array.shape) > 1:
        return [decode_tokens(tokenizer, row) for row in token_array]
    return [tokenizer.decode([t]) for t in token_array]


from typing import Any, Optional

import transformers


def find_token_range(
    string: str,
    substring: str,
    tokenizer: Optional[transformers.AutoTokenizer] = None,
    occurrence: int = 0,
    offset_mapping: Optional[torch.Tensor] = None,
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

    # print(char_start, char_end)

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


def guess_subject(prompt):
    return re.search(r"(?!Wh(o|at|ere|en|ich|y) )([A-Z]\S*)(\s[A-Z][a-z']*)*", prompt)[
        0
    ].strip()


from typing import Literal


def get_model_size(
    model: torch.nn.Module, unit: Literal["B", "KB", "MB", "GB"] = "MB"
) -> float:
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all = param_size + buffer_size
    denom = {"B": 1, "KB": 2**10, "MB": 2**20, "GB": 2**30}[unit]
    return size_all / denom
