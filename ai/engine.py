import abc

import aiohttp
from llama_cpp import Llama


# TODO: Samplers
# TODO: CFG
# TODO: Compression: https://github.com/microsoft/LLMLingua

class LLMEngine:
    @abc.abstractmethod
    def __init__(self, model, chat_format, n_ctx, verbose=False):
        pass

    @abc.abstractmethod
    async def complete(self):
        pass

    @abc.abstractmethod
    async def chat(self, chatlog: list[dict[str, str]], **kwargs):
        pass


class LlamaCppServer(LLMEngine):
    def __init__(self, model, chat_format="chatml", n_ctx=16 * 1024):
        self.llm = aiohttp.ClientSession(raise_for_status=True)
        self.chat_format = chat_format
        self.model = model  # TODO: Assert
        self.n_ctx = n_ctx


    async def complete(self):
        pass

    async def chat(self, chatlog: list[dict[str, str]], max_tokens=400, grammar=None, **kwargs):
        async with self.llm.post("/v1/chat/completions", json={'messages': chatlog, 'n_predict': max_tokens, **kwargs}) as resp:
            return (await resp.json())['choices'][0]['message']


class LlamaCppPython(LLMEngine):
    def __init__(self, model, chat_format="chatml", n_ctx=16 * 1024):
        self.llm = Llama(model_path=model, n_gpu_layers=99, n_ctx=n_ctx, chat_format=chat_format, verbose=False)

    async def complete(self):
        pass

    async def chat(self, chatlog, **kwargs):
        return self.llm.create_chat_completion(chatlog, **kwargs)['choices'][0]['message']


class ExLlama(LLMEngine):
    async def complete(self):
        pass

    async def chat(self, chatlog, **kwargs):
        pass

