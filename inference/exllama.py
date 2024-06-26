import logging
import random
import re
import sys
from functools import partial, cache
from itertools import takewhile
from typing import override

import anyio
from anyio import to_thread, EndOfStream
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream

from exllamav2 import ExLlamaV2Config, ExLlamaV2, ExLlamaV2Tokenizer, ExLlamaV2Cache_8bit
from exllamav2.generator import ExLlamaV2Sampler, ExLlamaV2StreamingGenerator
from lmformatenforcer import RegexParser
from lmformatenforcer.integrations.exllamav2 import ExLlamaV2TokenEnforcerFilter

from inference.engine import LLMEngine

_logger = logging.getLogger(__name__)


class ExLlama(LLMEngine):
    receive_stream: MemoryObjectReceiveStream[dict[str, str]]
    send_stream: MemoryObjectSendStream[dict[str, str]]
    BATCH_SIZE = 4
    QUEUE_SIZE = BATCH_SIZE * 16
    MAX_PAD_PERCENT = 0.2

    class _WouldOverflow(Exception):
        pass

    def __init__(self, model, n_ctx=16 * 1024):
        super().__init__()
        self.semaphore = anyio.Semaphore(1)  # Not thread safe.
        self.mistral = False
        config = ExLlamaV2Config()
        config.model_dir = model
        config.max_seq_len = n_ctx  # TODO: 16k is probably too high for Mistral without rope or sliding attention.
        config.max_batch_size = self.BATCH_SIZE
        config.prepare()
        exllama = ExLlamaV2(config)
        cache = ExLlamaV2Cache_8bit(exllama, lazy=True, batch_size=self.BATCH_SIZE, max_seq_len=n_ctx)
        exllama.load_autosplit(cache)
        self.tokenizer = ExLlamaV2Tokenizer(config)
        self.generator = ExLlamaV2StreamingGenerator(exllama, cache, self.tokenizer)
        self.generator.set_stop_conditions([self.tokenizer.eos_token_id, "<|im_end|>"])
        self.generator.warmup()
        self.send_stream, self.receive_stream = (
            anyio.create_memory_object_stream[dict[str, str]](max_buffer_size=self.QUEUE_SIZE))
        self.task_group = anyio.create_task_group()
        self.max_seq_len = cache.max_seq_len
        try:
            self.name = model.split("/")[-1].split("_")[-1]
            # self.name = "".join(model.split("/")[-1].split("_")[1:])
        except Exception as e:
            _logger.warning(f"Failed to parse model's name: {e}")

    async def complete(self, text, max_tokens=400, temperature=0.85, grammar: LLMEngine.Grammar = None, **kwargs):
        settings = ExLlamaV2Sampler.Settings()
        settings.top_k = 0
        settings.temperature = temperature
        settings.token_repetition_penalty = 1
        settings.min_p = 0.15
        settings.filters = [grammar] if grammar is not None else []
        settings.max_tokens = max_tokens
        settings.unique = 0
        done = anyio.Event()
        job = {"text": text, "settings": settings, "event": done}
        if settings.filters:
            # Filters not implemented for batch size > 1 in exllama
            settings.unique = random.randint(0, sys.maxsize)
        if self.BATCH_SIZE == 1:
            gen = partial(self.generator.generate_simple, encode_special_tokens=True, decode_special_tokens=True)
            async with self.semaphore:
                job["output"] = await to_thread.run_sync(gen, job["text"], job["settings"], job["settings"].max_tokens)
        else:
            # TODO: Continuous batching: https://github.com/turboderp/exllamav2/issues/95#issuecomment-1786542293
            #  https://github.com/epolewski/EricLLM/blob/main/ericLLM.py
            await self.send_stream.send(job)
            await done.wait()
        # Remove paddings
        return re.match(f'^(?:{self.tokenizer.pad_token})*(.*?)(?:{self.tokenizer.eos_token})*$', job["output"],
                        flags=re.DOTALL).group(1)

    async def chat(self, chatlog: list[dict[str, str]], **kwargs):
        if self.mistral:
            prompt = "<s>"
            for chat in chatlog:
                if chat["role"] == "user":
                    prompt += "[INST] " + chat["content"] + "[/INST]"
                else:
                    prompt += chat["content"] + "</s>"
        else:
            prompt = "".join(f"<|im_start|>{i["role"]}\n{i['content']}<|im_end|>\n" for i in chatlog)
            prompt += "<|im_start|>assistant\n"
        output = await self.complete(prompt, **kwargs)
        return {"role": "assistant", "content": output[len(prompt):].strip()}

    async def run(self):
        gen = partial(self.generator.generate_simple, encode_special_tokens=True, decode_special_tokens=True)
        incompatibles = []
        while True:
            try:
                jobs = [incompatibles.pop() if incompatibles else await self.receive_stream.receive()]
            except EndOfStream:
                return
            new_incompatibles = []
            while len(jobs) < self.BATCH_SIZE and jobs[0]["settings"].unique == 0:  # Don't try to batch forced unique.
                try:
                    cantidate_job = incompatibles.pop() if incompatibles else self.receive_stream.receive_nowait()
                    if self.can_batch(jobs, cantidate_job):
                        jobs.append(cantidate_job)
                    else:
                        new_incompatibles.append(cantidate_job)
                except (anyio.WouldBlock, self._WouldOverflow):
                    break
            incompatibles.extend(new_incompatibles)  # Done after to prevent an infinite loop
            if len(jobs) == self.BATCH_SIZE:
                _logger.info(f"Batching {len(jobs)} jobs.")
                outputs = await to_thread.run_sync(gen, [job["text"] for job in jobs], jobs[0]["settings"],
                                                   jobs[0]["settings"].max_tokens)
                for output, job in zip(outputs, jobs):
                    job["output"] = output
                    job["event"].set()
            else:
                for job in jobs:
                    job["output"] = await to_thread.run_sync(gen, job["text"], job["settings"],
                                                             job["settings"].max_tokens)
                    job["event"].set()

    async def __aenter__(self):
        await self.task_group.__aenter__()
        if self.BATCH_SIZE > 1:
            self.task_group.start_soon(self.run)
        return self

    async def __aexit__(self, *args, **kwargs):
        await self.send_stream.aclose()
        await self.receive_stream.aclose()
        await self.task_group.__aexit__(*args, **kwargs)

    def can_batch(self, current_jobs: list[dict], candidate_job: dict) -> bool:
        setting1 = current_jobs[0]["settings"]
        setting2 = candidate_job["settings"]
        # Exllama doesn't support generation when settings differ.
        if not (setting1.filters == setting2.filters and
                setting1.max_tokens == setting2.max_tokens and
                setting1.min_p == setting2.min_p and
                setting1.mirostat == setting2.mirostat and
                setting1.mirostat_eta == setting2.mirostat_eta and
                setting1.mirostat_mu == setting2.mirostat_mu and
                setting1.mirostat_tau == setting2.mirostat_tau and
                setting1.temperature == setting2.temperature and
                setting1.temperature_last == setting2.temperature_last and
                setting1.tfs == setting2.tfs and
                setting1.token_bias == setting2.token_bias and
                setting1.token_repetition_decay == setting2.token_repetition_decay and
                setting1.token_repetition_penalty == setting2.token_repetition_penalty and
                setting1.token_repetition_range == setting2.token_repetition_range and
                setting1.top_a == setting2.top_a and
                setting1.top_k == setting2.top_k and
                setting1.top_p == setting2.top_p and
                setting1.typical == setting2.typical and
                setting1.unique == setting2.unique):
            return False
        # TODO: Reuse tokens.
        tokens = self.tokenizer.encode([job["text"] for job in current_jobs + [candidate_job]],
                                       encode_special_tokens=True)
        pad_token = self.tokenizer.pad_token_id
        # Compute the ratio of the sum of all front paddings to the total number of tokens.
        worst_pad_ratio = sum(len(list(takewhile(lambda x: x == pad_token, i))) for i in tokens) / tokens.numel()
        # Would be inefficient.
        if worst_pad_ratio > self.MAX_PAD_PERCENT:
            return False

        # This calculation is taken from exllama's generate_simple.
        if tokens.shape[-1] + current_jobs[0]["settings"].max_tokens > self.max_seq_len:
            # It's unlikely another job will be small enough to not overflow but big enough for the similarity criterion
            raise self._WouldOverflow()

        return True

    @override
    @cache
    def get_options_grammar(self, options: tuple[str, ...]) -> LLMEngine.Grammar:
        return ExLlamaV2TokenEnforcerFilter(RegexParser('|'.join(f'({option})' for option in options)), self.tokenizer)
