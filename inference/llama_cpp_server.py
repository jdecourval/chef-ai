import logging
import os
from functools import cache
from typing import override

import aiohttp
import anyio
import sh
from aiohttp import ClientTimeout

from inference.engine import LLMEngine

_logger = logging.getLogger(__name__)


class LlamaCppServer(LLMEngine):
    def __init__(self, model, n_ctx=10 * 1024):
        super().__init__()
        parallel = 3  # More than 3, with current settings, cause OOM.
        quantized_k_cache = False  # Saves a bit of VRAM, but currently result in a token generation slowdown.
        llama_server = sh.Command(f'{os.environ["LLAMACPP_BIN_DIR"]}/server')
        self.llama_server = llama_server('--model', model, '--n-gpu-layers', 99, '--ctx_size', n_ctx * parallel,
                                         '--parallel', parallel, '--cont-batching',
                                         '--cache-type-k', 'q8_0' if quantized_k_cache else 'f16',
                                         _bg=True, _out="llamacpp.log", _err="llamacpp.err.log")
        self.llama_client = aiohttp.ClientSession(raise_for_status=True, base_url="http://127.0.0.1:8080",
                                                  timeout=ClientTimeout(sock_read=600))
        self.semaphore = anyio.Semaphore(parallel * 10)
        try:
            self.name = "".join(model.split("/")[-1].split(".")[:-2])
        except Exception as e:
            _logger.warning(f"Failed to parse model's name: {e}ne")

    async def aclose(self):
        self.llama_server.terminate()
        await self.llama_client.close()

    async def __aenter__(self):
        # Make sure the server is ready
        while True:
            try:
                async with self.llama_client.get("/health", timeout=ClientTimeout(connect=600)) as resp:
                    if (await resp.json())["status"] == "ok":
                        return self
            except aiohttp.ClientError:
                pass
            await anyio.sleep(1)

    async def complete(self, text, **kwargs):
        pass

    async def chat(self, chatlog: list[dict[str, str]], **kwargs):
        async with self.semaphore:
            async with self.llama_client.post("/v1/chat/completions",
                                              json={'messages': chatlog, 'max_tokens': 400, **kwargs}) as resp:
                return (await resp.json())['choices'][0]['message']

    @override
    @cache
    def get_options_grammar(self, options: tuple[str, ...]) -> LLMEngine.Grammar:
        return f"root ::= {' | '.join(f'"{option}"' for option in options)}"
