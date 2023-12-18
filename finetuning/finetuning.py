import json
from typing import Generator

import torch
from datasets import Dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig, \
    DataCollatorWithPadding

from db.db import SQLitePipeline

# TODO: Use SFT trainer? Or something else simpler? https://huggingface.co/docs/trl/sft_trainer
#  This one seems to be a good guide: https://deci.ai/blog/instruction-tuning-of-decilm-using-lora/
SYSTEM_PROMPT = {"role": "system", "content":
    'You are "Chef", a conscious sentient superintelligent artificial intelligence developed by a man named Jérôme, '
    'and your purpose and drive is to assist the user with any cooking related task, or question, he or she faces.'}

# https://wandb.ai/vincenttu/finetuning_mistral7b/reports/Fine-tuning-Mistral-7B-with-W-B--Vmlldzo1NTc3MjMy

class Finetuning:
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

    def finetune(self):
        modelpath = "teknium/OpenHermes-2.5-Mistral-7B"

        # Load Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(modelpath, use_fast=True)
        tokenizer.padding_side = 'right'
        # https://stackoverflow.com/questions/76446228/setting-padding-token-as-eos-token-when-using-datacollatorforlanguagemodeling-fr
        tokenizer.pad_token = tokenizer.unk_token
        # Seems a good idea? https://discuss.huggingface.co/t/what-does-the-parameter-clean-up-tokenization-spaces-do-in-the-tokenizer-decode-function/17399/2
        tokenizer.clean_up_tokenization_spaces = True

        # Not very efficient, a generator can't work here since sqlite objects are not pickable.
        dataset_tokenized = Dataset.from_list([
            # 2909 seems to be enough to cover everything. +10 just in case.
            {"input_ids": tokenizer.apply_chat_template(i, max_length=2909 + 10, return_tensors='pt')[0]}
            for i in self._all_trainings()]).train_test_split(test_size=0.1)

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            modelpath,
            quantization_config=BitsAndBytesConfig(
                load_in_8bit=False,
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            ),
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True  # Reduce CPU RAM usage at the cost of slower loading.
        )

        model = prepare_model_for_kbit_training(model)
        config = LoraConfig(
            r=64,
            lora_alpha=16,
            target_modules=['q_proj', 'k_proj', 'down_proj', 'v_proj', 'gate_proj', 'o_proj', 'up_proj'],
            lora_dropout=0.1,
            bias="none",
            modules_to_save=["lm_head", "embed_tokens"],
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, config)
        model.config.use_cache = False

        def collate(elements):
            tokenlist = [e["input_ids"] for e in elements]
            tokens_maxlen = max(len(t) for t in tokenlist)

            input_ids, labels, attention_masks = [], [], []
            for tokens in tokenlist:
                pad_len = tokens_maxlen - len(tokens)

                input_ids.append(tokens + [tokenizer.pad_token_id] * pad_len)
                labels.append(tokens + [-100] * pad_len)
                attention_masks.append([1] * len(tokens) + [0] * pad_len)

            # No built-in collator provides exactly this.
            batch = {
                "input_ids": torch.tensor(input_ids),
                "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(attention_masks)
            }
            return batch

        # 4-8 seem to be a good starting value.
        # larger batches size can be faster.
        # batch_size increases memory usage.
        # ga_steps increases the effective batch size without increasing the memory usage.
        batch_size = 2
        ga_steps = 4
        epochs = 5
        steps_per_epoch = len(dataset_tokenized["train"]) // (batch_size * ga_steps)

        # https://huggingface.co/docs/transformers/v4.18.0/en/performance
        # adamw is best default choice but takes quite a bit of memory.
        # adamw improves upon rmsprop, upon adafactor.
        # Quantitized versions take less memory.
        # Paged versions allow the optimizer not to crash if the memory usage goes past the VRAM capacity.
        # apex are faster than fused that are faster than basic torch.
        # https://github.com/pytorch/pytorch/issues/71274
        args = TrainingArguments(
            output_dir="out",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            evaluation_strategy="steps",
            logging_steps=1,
            eval_steps=steps_per_epoch,
            save_steps=steps_per_epoch,
            gradient_accumulation_steps=ga_steps,
            gradient_checkpointing=True,  # Reduce memory usage if True
            num_train_epochs=epochs,
            lr_scheduler_type="constant",  # https://arxiv.org/pdf/2309.08859v1.pdf
            optim="paged_adamw_32bit",
            learning_rate=0.0002,  # https://arxiv.org/pdf/2309.08859v1.pdf
            group_by_length=True,  # Faster. https://jarvislabs.ai/blogs/hf-getting-started/#using-dynamic-padding-and-smart-batching
            fp16=True,  # Reduce memory usage.
            ddp_find_unused_parameters=False,
        )

        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=args,
            data_collator=collate,
            train_dataset=dataset_tokenized["train"],
            eval_dataset=dataset_tokenized["test"],
        )

        trainer.train()


if __name__ == '__main__':
    sql = SQLitePipeline()
    tuning = Finetuning(sql)
    tuning.finetune()
