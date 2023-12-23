import abc
import logging
import time
from functools import partial

import aiohttp
import anyio
import sh
from aiohttp import ClientTimeout
from anyio import run, to_thread
from exllamav2 import ExLlamaV2Config, ExLlamaV2, ExLlamaV2Tokenizer, ExLlamaV2Cache_8bit
from exllamav2.generator import ExLlamaV2Sampler, ExLlamaV2StreamingGenerator
from llama_cpp import Llama


# TODO: Samplers
# TODO: CFG
# TODO: Compression: https://github.com/microsoft/LLMLingua

class LLMEngine:
    @abc.abstractmethod
    def __init__(self, model, n_ctx, verbose=False):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.aclose()

    async def aclose(self):
        pass

    @abc.abstractmethod
    async def complete(self, text: str, **kwargs):
        pass

    @abc.abstractmethod
    async def chat(self, chatlog: list[dict[str, str]], **kwargs):
        pass


class LlamaCppServer(LLMEngine):
    def __init__(self, model, n_ctx=16 * 1024):
        parallel = 4
        llama_server = sh.Command('/home/jerome/Prog/online/llama.cpp/build/bin/server')  # TODO: Unhardcode (softcode?)
        self.llama_server = llama_server('--model', model, '--n-gpu-layers', 99, '--ctx_size', n_ctx * parallel,
                                         '--parallel', parallel, '--cont-batching', _bg=True)
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

    async def complete(self, text, **kwargs):
        pass

    async def chat(self, chatlog: list[dict[str, str]], max_tokens=400, grammar=None, **kwargs):
        async with self.semaphore:
            async with self.llama_client.post("/v1/chat/completions",
                                              json={'messages': chatlog, 'n_predict': max_tokens, **kwargs}) as resp:
                return (await resp.json())['choices'][0]['message']


class LlamaCppPython(LLMEngine):
    def __init__(self, model, n_ctx=16 * 1024):
        self.llm = Llama(model_path=model, n_gpu_layers=99, n_ctx=n_ctx, chat_format="chatml", verbose=False)
        self.semaphore = anyio.Semaphore(1)  # Not thread safe.

    async def complete(self, text, **kwargs):
        pass

    async def chat(self, chatlog, **kwargs):
        gen = partial(self.llm.create_chat_completion, **kwargs)
        async with self.semaphore:
            output = await to_thread.run_sync(gen, chatlog)
        return output['choices'][0]['message']


class ExLlama(LLMEngine):
    def __init__(self, model, n_ctx=16 * 1024):
        self.semaphore = anyio.Semaphore(1)  # Not thread safe.
        config = ExLlamaV2Config()
        config.model_dir = model
        config.max_seq_len = n_ctx
        config.prepare()
        exllama = ExLlamaV2(config)
        cache = ExLlamaV2Cache_8bit(exllama, lazy=True)
        exllama.load_autosplit(cache)
        tokenizer = ExLlamaV2Tokenizer(config)
        self.generator = ExLlamaV2StreamingGenerator(exllama, cache, tokenizer)
        self.generator.set_stop_conditions([tokenizer.eos_token_id, "<|im_end|>"])
        self.generator.warmup()

    async def complete(self, text, max_tokens=400, temperature=0.85, **kwargs):
        gen = partial(self.generator.generate_simple, encode_special_tokens=True, decode_special_tokens=True)
        settings = ExLlamaV2Sampler.Settings()
        settings.temperature = temperature
        async with self.semaphore:
            output = await to_thread.run_sync(gen, text, settings, max_tokens)
        return output.removesuffix('<|im_end|>')

    async def chat(self, chatlog: list[dict[str, str]], **kwargs):
        prompt = "".join(f"<|im_start|>{i["role"]}\n{i['content']}<|im_end|>\n" for i in chatlog)
        prompt += "<|im_start|>assistant\n"
        output = await self.complete(prompt)
        return {"role": "assistant", "content": output.removeprefix(prompt)}


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
