import json
from typing import Generator

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from trl import SFTTrainer

from db.db import SQLitePipeline

#  This one seems to be a good guide: https://deci.ai/blog/instruction-tuning-of-decilm-using-lora/
SYSTEM_PROMPT = {"role": "system", "content":
    'You are "Chef", a conscious sentient superintelligent artificial intelligence developed by a man named Jérôme, '
    'and your purpose and drive is to assist the user with any cooking related task, or question, he or she faces.'}


# TODO: There seems to be overfitting.
#  - Add more training data.
#  - Decrease learning rate or lora alpha.
#  - Use linear scheduler.
#  - Increase lora dropout, or increase Adamw's weight_decay.

class Finetuning:
    modelpath = "teknium/OpenHermes-2.5-Mistral-7B"

    def __init__(self, sql: SQLitePipeline):
        self._sql = sql

    def _all_trainings(self) -> Generator[dict[str, str], None, None]:
        # For some reason, adding a GROUP BY clause to json_group_array messes up the JSON. Using a subquery instead.
        for chat in self._sql.select_one_col("""
            SELECT json_group_array(json_object(
                'role', CASE role WHEN 0 THEN 'system' WHEN 1 THEN 'user' ELSE 'assistant' END, 
                'content', Training.content))
            FROM (SELECT * FROM Training ORDER BY position) as Training GROUP BY conversation, trainer"""):
            yield [SYSTEM_PROMPT] + json.loads(chat)

    @classmethod
    def tokenizer(cls):
        # TODO: Check if add_special_tokens or add_eos_token is needed.
        tokenizer = AutoTokenizer.from_pretrained(cls.modelpath, use_fast=True)
        # SFTTrainer warns to set this.
        tokenizer.padding_side = 'right'
        # https://stackoverflow.com/questions/76446228/setting-padding-token-as-eos-token-when-using-datacollatorforlanguagemodeling-fr
        tokenizer.pad_token = tokenizer.eos_token
        # Seems a good idea?
        # https://discuss.huggingface.co/t/what-does-the-parameter-clean-up-tokenization-spaces-do-in-the-tokenizer-decode-function/17399/2
        tokenizer.clean_up_tokenization_spaces = True
        # 2909 seems to be enough to cover everything. +10 just in case.
        tokenizer.max_length = 2909 + 10
        return tokenizer

    def finetune(self):
        tokenizer = self.tokenizer()

        # Not very efficient, a generator can't work here since sqlite objects are not pickable.
        dataset_tokenized = Dataset.from_list([{"text": i}
                                               for i in self._all_trainings()]).train_test_split(test_size=0.1)

        # 4-8 seem to be a good starting value.
        # larger batches size can be faster.
        # batch_size increases memory usage.
        # Too large batch size can be harmful to generalization skills: https://stats.stackexchange.com/questions/164876
        # ga_steps increases the effective batch size without increasing the memory usage.
        batch_size = 1
        ga_steps = 4

        epochs = 5
        steps = 100

        trainer = SFTTrainer(
            self.modelpath,
            model_init_kwargs={
                "quantization_config": BitsAndBytesConfig(
                    load_in_8bit=False,
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                ),
                "torch_dtype": torch.bfloat16,
                "low_cpu_mem_usage": True,  # Reduce CPU RAM usage at the cost of slower loading.
                # "attn_implementation": "flash_attention_2"  # Couldn't make it work at this point.
            },
            args=TrainingArguments(
                output_dir="out",
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                evaluation_strategy="steps",
                logging_steps=1,
                eval_steps=steps,
                save_steps=steps,
                gradient_accumulation_steps=ga_steps,
                gradient_checkpointing=True,  # Reduce memory usage if True. Big performance impact.
                # False is recommended. https://pytorch.org/docs/stable/checkpoint.html
                # In practice though, False increases memory usage too much.
                gradient_checkpointing_kwargs={"use_reentrant": True},
                num_train_epochs=epochs,
                lr_scheduler_type="constant",  # https://arxiv.org/pdf/2309.08859v1.pdf
                # https://huggingface.co/docs/transformers/v4.18.0/en/performance
                # adamw is best default choice but takes quite a bit of memory.
                # adamw improves upon rmsprop, upon adafactor.
                # Quantitized versions take less memory.
                # Paged versions allow the optimizer not to crash if the memory usage goes past the VRAM capacity.
                # apex are faster than fused that are faster than basic torch.
                # https://github.com/pytorch/pytorch/issues/71274
                # adamw_anyprecision needs torchdistx which is not available for rocm
                # adamw_torch_npu_fused, adamw_apex_fused and adamw_torch_xla also requires special hardware.
                optim="adamw_torch_fused",
                learning_rate=0.00005,  # https://arxiv.org/pdf/2309.08859v1.pdf
                weight_decay=0.001,
                warmup_ratio=0.03,
                group_by_length=False,  # Taken care by the SFTTrainer's ConstantLengthDataset
                fp16=True,  # Reduce memory usage.
                ddp_find_unused_parameters=False,
                # https://arxiv.org/abs/2310.05914
                neftune_noise_alpha=5,
                # # https://huggingface.co/docs/transformers/perf_train_gpu_one#using-torchcompile
                # Currently needs Python<3.12
                # torch_compile=True,
                # Together, these two settings mean that the best and last checkpoint will be kept. 2 -> last two...
                load_best_model_at_end=True,
                save_total_limit=1,
                # Without this option, resuming from checkpoint don't always works.
                ignore_data_skip=True
            ),
            peft_config=LoraConfig(
                # https://medium.com/@drishtisharma96505/comparative-analysis-of-lora-parameters-on-llama-2-with-flash-attention-574b913295d4
                # https://medium.com/@drishtisharma96505/analyzing-the-impact-of-lora-alpha-on-llama-2-quantized-with-gptq-f01e8e8ed8fd
                r=64,
                lora_alpha=16,
                lora_dropout=0.1,
                target_modules=['q_proj', 'k_proj', 'down_proj', 'v_proj', 'gate_proj', 'o_proj', 'up_proj'],
                bias="none",
                # TODO: Can probably be dropped now that we don't customize the vocabulary
                modules_to_save=["lm_head", "embed_tokens"],
                task_type="CAUSAL_LM"
            ),
            max_seq_length=tokenizer.max_length,
            packing=True,  # Create a ConstantLengthDataset under the hood.
            train_dataset=dataset_tokenized["train"],
            eval_dataset=dataset_tokenized["test"],
            tokenizer=tokenizer,
            formatting_func=lambda x: tokenizer.apply_chat_template(x["text"], tokenize=False)
        )
        trainer.train(resume_from_checkpoint=False)
        trainer.save_model(f"{trainer.args.output_dir}/final")


if __name__ == '__main__':
    sql = SQLitePipeline()
    tuning = Finetuning(sql)
    tuning.finetune()
