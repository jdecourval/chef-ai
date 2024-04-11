import logging
from functools import partial, cache
from typing import override

import anyio
from anyio import to_thread
from llama_cpp import Llama, LlamaGrammar

from inference.engine import LLMEngine

_logger = logging.getLogger(__name__)


class LlamaCppPython(LLMEngine):
    def __init__(self, model, n_ctx=16 * 1024):
        super().__init__()
        self.llm = Llama(model_path=model, n_gpu_layers=99, n_ctx=n_ctx, chat_format="chatml", offload_kqv=True,
                         verbose=False)
        self.semaphore = anyio.Semaphore(1)  # Not thread safe.
        try:
            self.name = "".join(model.split("/")[-1].split(".")[:-2])
        except Exception as e:
            _logger.warning(f"Failed to parse model's name: {e}")

    async def complete(self, text, **kwargs):
        pass

    async def chat(self, chatlog, **kwargs):
        gen = partial(self.llm.create_chat_completion, **kwargs)
        async with self.semaphore:
            output = await to_thread.run_sync(gen, chatlog)
        return output['choices'][0]['message']

    @override
    @cache
    def get_options_grammar(self, options: tuple[str, ...]) -> LLMEngine.Grammar:
        return LlamaGrammar.from_string(f"root ::= {' | '.join(f'"{option}"' for option in options)}", verbose=False)
