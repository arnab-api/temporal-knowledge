import re
from typing import Literal, Optional

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
