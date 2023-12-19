from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Source: https://github.com/geronimi73/qlora-minimal/tree/main
base_path = "mistralai/Mistral-7B-v0.1"  # input: base model
adapter_path = "out/checkpoint-33635"  # input: adapters
save_to = "out/merged"  # out: merged model ready for inference

base_model = AutoModelForCausalLM.from_pretrained(
    base_path,
    return_dict=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(base_path)
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.unk_token
tokenizer.clean_up_tokenization_spaces = True
tokenizer.max_length = 2909 + 10

# Load LoRA adapter and merge
model = PeftModel.from_pretrained(base_model, adapter_path)
model = model.merge_and_unload()

model.save_pretrained(save_to, safe_serialization=True, max_shard_size='4GB')
tokenizer.save_pretrained(save_to)
