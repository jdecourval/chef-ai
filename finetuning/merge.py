from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
from finetuning.finetuning import Finetuning

# Source: https://github.com/geronimi73/qlora-minimal/tree/main
base_path = "teknium/OpenHermes-2.5-Mistral-7B"  # input: base model
adapter_path = "out/checkpoint-842"  # input: adapters
save_to = "out/merged-v4"  # out: merged model ready for inference

base_model = AutoModelForCausalLM.from_pretrained(
    base_path,
    return_dict=True,
    torch_dtype=torch.bfloat16,  # float16?
    device_map="auto",
)

tokenizer = Finetuning.tokenizer()

# Load LoRA adapter and merge
model = PeftModel.from_pretrained(base_model, adapter_path)
model = model.merge_and_unload()

model.save_pretrained(save_to, safe_serialization=True, max_shard_size='4GB')
tokenizer.save_pretrained(save_to)
