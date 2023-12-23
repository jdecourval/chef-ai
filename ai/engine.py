import abc
import logging
import time

import aiohttp
import anyio
import sh
from aiohttp import ClientTimeout
from anyio import run
from llama_cpp import Llama


# TODO: Samplers
# TODO: CFG
# TODO: Compression: https://github.com/microsoft/LLMLingua

class LLMEngine:
    @abc.abstractmethod
    def __init__(self, model, chat_format, n_ctx, verbose=False):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.aclose()

    async def aclose(self):
        pass

    @abc.abstractmethod
    async def complete(self):
        pass

    @abc.abstractmethod
    async def chat(self, chatlog: list[dict[str, str]], **kwargs):
        pass


class LlamaCppServer(LLMEngine):
    def __init__(self, model, chat_format="chatml", n_ctx=16 * 1024):
        parallel = 4
        llama_server = sh.Command('/home/jerome/Prog/online/llama.cpp/build/bin/server')  # TODO: Unhardcode (softcode?)
        self.llama_server = llama_server('--model', model, '--n-gpu-layers', 99, '--ctx_size', n_ctx * parallel, '--parallel',
                                         parallel, '--cont-batching', _bg=True, _out="/tmp/output")
        self.llama_client = aiohttp.ClientSession(raise_for_status=True, base_url="http://127.0.0.1:8080",
                                                  timeout=ClientTimeout(sock_read=600))
        # Increasing this leds to slowdown on pure token-generation work loads.
        self.semaphore = anyio.Semaphore(parallel)

    async def aclose(self):
        self.llama_server.terminate()
        await self.llama_client.close()

    async def __aenter__(self):
        # Make sure the server is ready
        while True:
            try:
                async with self.llama_client.get("/", timeout=ClientTimeout(connect=600)):
                    return self
            except aiohttp.ClientConnectorError:
                await anyio.sleep(1)

    async def complete(self):
        pass

    async def chat(self, chatlog: list[dict[str, str]], max_tokens=400, grammar=None, **kwargs):
        async with self.semaphore:
            async with self.llama_client.post("/v1/chat/completions",
                                              json={'messages': chatlog, 'n_predict': max_tokens, **kwargs}) as resp:
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


if __name__ == '__main__':
    async def main():
        async with LlamaCppServer(
            '/home/jerome/Prog/online/oobabooga_linux/text-generation-webui/models/openhermes-2.5-mistral-7b-16k.Q5_K_M.gguf') as llama:
            async def test(query):
                print(await llama.chat([{"role": "user", "content": query}]))

            now = time.monotonic()
            # The goal is just to have something long to exercise prompt processing.
            prefix = ("You are a helpful assistant that answers all questions without hesitation. You must be "
                      "reasonably sure that your answers are correct, otherwise you must respond with 'I am sorry, "
                      "but I don't know the answer to that question.'.")
            async with anyio.create_task_group() as tg:
                tg.start_soon(test, prefix + "What is the meaning of life?")
                tg.start_soon(test, prefix + "What is the meaning of love?")
                tg.start_soon(test, prefix + "What is the meaning of Breathe, by Pink Floyd?")
                tg.start_soon(test, prefix + "What is the meaning of Jesus?")
                tg.start_soon(test, prefix + "What is the meaning of death?")
                tg.start_soon(test, prefix + "What is the meaning of Easter bunny?")

            print(time.monotonic() - now)

    logging.basicConfig(level=logging.INFO)
    run(main)
