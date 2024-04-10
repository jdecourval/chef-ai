# Source: https://twitter.com/Tim_Dettmers/status/1695352747694919931
import copy
import json
import os

import bitsandbytes as bnb
import torch
from bitsandbytes.functional import dequantize_4bit
from peft import PeftModel
from peft.utils import _get_submodules
from transformers import LlamaForCausalLM


def _save_model(model, tokenizer, to):
    model.save_pretrained(to)
    tokenizer.save_pretrained(to)
    config_data = json.loads(open(os.path.join(to, 'config.json'), 'r').read())
    config_data.pop("quantization_config", None)
    config_data.pop("pretraining_tp", None)
    with open(os.path.join(to, 'config.json'), 'w') as config:
        config.write(json.dumps(config_data, indent=2))


def _dequantize_model(model, dtype=torch.bfloat16):
    """
    'model': the peftmodel you loaded with qlora.
    'dtype': dtype that the model was trained using
    'device': device to load the model to
    """

    cls = bnb.nn.Linear4bit

    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, cls):
                print(f"Dequantizing `{name}`...")
                quant_state = copy.deepcopy(module.weight.quant_state)

                # quant_state[2] = dtype
                quant_state.dtype = dtype

                weights = dequantize_4bit(module.weight.data, quant_state=quant_state, quant_type="nf4").to(dtype)

                new_module = torch.nn.Linear(module.in_features, module.out_features, bias=False, dtype=dtype)
                new_module.weight = torch.nn.Parameter(weights)

                parent, target, target_name = _get_submodules(model, name)
                setattr(parent, target_name, new_module)

        model.is_loaded_in_4bit = False
        return model


def merge_lora(base_model, adapter_path, output, tokenizer, model_init_kwargs):
    model = LlamaForCausalLM.from_pretrained(base_model, **model_init_kwargs, device_map="auto")
    model = _dequantize_model(model, model_init_kwargs["torch_dtype"])
    model = PeftModel.from_pretrained(model=model, model_id=adapter_path)
    model = model.merge_and_unload(progressbar=True)
    _save_model(model, tokenizer, output)
