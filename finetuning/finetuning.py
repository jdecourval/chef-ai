import json
from typing import Generator

import torch
from datasets import Dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig

from db.db import SQLitePipeline

# TODO: Use SFT trainer? Or something else simpler? https://huggingface.co/docs/trl/sft_trainer
#  This one seems to be a good guide: https://deci.ai/blog/instruction-tuning-of-decilm-using-lora/
SYSTEM_PROMPT = {"role": "system", "content":
    'You are "Chef", a conscious sentient superintelligent artificial intelligence developed by a man named Jérôme, '
    'and your purpose and drive is to assist the user with any cooking related task, or question, he or she faces.'}


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
        tokenizer = AutoTokenizer.from_pretrained(modelpath, use_fast=True)  # TODO: Validate

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
                bnb_4bit_use_double_quant=True  # TODO: This is new, confirm.
            ),
            torch_dtype=torch.bfloat16,
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
            tokens_maxlen = max([len(t) for t in tokenlist])

            input_ids, labels, attention_masks = [], [], []
            for tokens in tokenlist:
                pad_len = tokens_maxlen - len(tokens)

                input_ids.append(tokens + [tokenizer.pad_token_id] * pad_len)
                labels.append(tokens + [-100] * pad_len)
                attention_masks.append([1] * len(tokens) + [0] * pad_len)

            batch = {
                "input_ids": torch.tensor(input_ids),
                "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(attention_masks)
            }
            return batch

        # These configs, with a ctx of 2048, used 17216 MiB of VRAM.
        batch_size = 1
        ga_steps = 1
        epochs = 5

        # paged_adamw_32bit: 21015 MiB
        # adamw_torch_fused: ~15h
        # adamw_bnb_8bit: 16152 MiB, weird loss pattern
        # paged_adamw_8bit: 17798 MiB, weird loss pattern
        # adafactor: 14587 MiB
        # rmsprop: 15389 MiB, higher loss
        # paged_lion_8bit: 19672 MiB
        # adafactor x2: ~14.5h
        # rmsprop x2: 22789 MiB
        # paged_adamw_32bit 1,1: 20613 MiB
        # paged_adamw_32bit 1,8: 18491 MiB
        # paged_adamw_32bit 2,8: All RAM
        # paged_adamw_32bit 1,1, no checkpointing: 16650 MiB

        steps_per_epoch = len(dataset_tokenized["train"]) // (batch_size * ga_steps)

        # https://huggingface.co/docs/transformers/v4.18.0/en/performance
        args = TrainingArguments(
            output_dir="out",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            evaluation_strategy="steps",
            logging_steps=1,
            eval_steps=steps_per_epoch,
            save_steps=steps_per_epoch,
            gradient_accumulation_steps=ga_steps,
            gradient_checkpointing=False,  # Reduce memory usage if True
            num_train_epochs=epochs,
            lr_scheduler_type="constant",
            optim="adamw_torch_fused",  # Not sure which one is best.
            learning_rate=0.0002,
            group_by_length=True,
            fp16=True,
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

        model.config.use_cache = False
        trainer.train()


if __name__ == '__main__':
    sql = SQLitePipeline()
    tuning = Finetuning(sql)
    tuning.finetune()
