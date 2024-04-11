import abc
import logging
import time
from typing import TypeVar

import anyio
from anyio import run

# TODO: Samplers
# TODO: CFG
# TODO: Compression: https://github.com/microsoft/LLMLingua


_logger = logging.getLogger(__name__)


class LLMEngine:
    Grammar = TypeVar('Grammar')

    @abc.abstractmethod
    def __init__(self):
        self.name = ""

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

    def model_name(self):
        return self.name

    @abc.abstractmethod
    def get_options_grammar(self, options: tuple[str, ...]) -> Grammar:
        pass


if __name__ == '__main__':
    import argparse
    from inference.llama_cpp_server import LlamaCppServer
    from inference.exllama import ExLlama

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    args = parser.parse_args()


    async def main():
        async with LlamaCppServer(args.model) if args.model.endswith(".gguf") else ExLlama(args.model) as llama:
            async def test(query):
                print(await llama.chat([{"role": "user", "content": query}], max_tokens=1000))

            await test("warmup")

            now = time.monotonic()
            # The goal is just to have something long to exercise prompt processing.
            prefix = ("You are a helpful assistant that answers all questions without hesitation. You must be "
                      "reasonably sure that your answers are correct, otherwise you must respond with 'I am sorry, "
                      "but I don't know the answer to that question.'. ")
            async with anyio.create_task_group() as tg:
                for _ in range(3):
                    tg.start_soon(test, prefix + "What is the meaning of life?")
                    tg.start_soon(test, prefix + "What is the meaning of love?")
                    tg.start_soon(test, prefix + "What is the meaning of Breathe, by Pink Floyd?")
                    tg.start_soon(test, prefix + "What is the meaning of Jesus?")
                    tg.start_soon(test, prefix + "What is the meaning of death?")
                    tg.start_soon(test, prefix + "What is the meaning of Easter bunny?")

            print(time.monotonic() - now)


    logging.basicConfig(level=logging.INFO)
    run(main)
