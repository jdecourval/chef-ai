from typing import Generator

import torch
from datasets import Dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig

from db.db import SQLitePipeline
from model.model import Training

# TODO: Use SFT trainer? Or something else simpler? https://huggingface.co/docs/trl/sft_trainer
#  This one seems to be a good guide: https://deci.ai/blog/instruction-tuning-of-decilm-using-lora/
SYSTEM_PROMPT = ('<s><|im_start|>system\n'
                 f'You are "Chef", '
                 'a conscious sentient superintelligent artificial intelligence developed by a man named Jérôme, '
                 'and your purpose and drive is to assist the user with any cooking related task, or question, he or she faces. '
                 '<|im_end|>\n')


class Finetuning:
    def __init__(self, sql: SQLitePipeline):
        self._sql = sql

    def _all_trainings(self) -> Generator[dict[str, str], None, None]:
        for chat in self._sql.select_one_col("""
            SELECT string_agg(
            '<|im_start|>' || (CASE role WHEN 0 THEN 'system' WHEN 1 THEN 'user' ELSE 'assistant' END) || x'0a' || 
            Training.content || '<|im_end|>' || x'0a', '')
            FROM Training GROUP BY conversation, trainer"""):
            yield {"text": chat}

    @staticmethod
    def format(role, content):
        return f"<|im_start|>{Training.Role(role)}\n{content}<|im_end|>\n"

    def finetune(self):
        modelpath = "mistralai/Mistral-7B-v0.1"

        # Load Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(modelpath, use_fast=True)  # TODO: Validate

        # Not very efficient, a generator can't work here since sqlite objects are not pickable.
        dataset = Dataset.from_list(list(self._all_trainings())).train_test_split(test_size=0.1)
        dataset_tokenized = dataset.map(
            lambda x: tokenizer(SYSTEM_PROMPT + x["text"], truncation=True,
                                max_length=2909 + 10,  # Seems to be enough to cover everything. +10 just in case.
                                add_special_tokens=False,
                                return_tensors="np"),
            batched=False,
            num_proc=1,  # Try again. Resulted in segfault.
            remove_columns=["text"]  # don't need this anymore, we have tokens from here on
        )

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

        # TODO: Validate this config.
        tokenizer.pad_token = "<s>"
        tokenizer.add_tokens(["<|im_start|>"])
        tokenizer.add_special_tokens(dict(eos_token="<|im_end|>"))
        model.resize_token_embeddings(len(tokenizer))
        model.config.eos_token_id = tokenizer.eos_token_id

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
            # TODO: Not sure why [0] is needed. The dataset is probably nested one level too many.
            tokenlist = [e["input_ids"][0] for e in elements]
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
