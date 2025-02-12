import logging
from contextlib import contextmanager

from inference.engine import LLMEngine

_logger = logging.getLogger(__name__)


class Chat:
    DEFAULT_SYSTEM_PROMPT = ("You are a helpful assistant. Below is an instruction that describes a task. "
                             "Write a response that appropriately completes the request.")

    def __init__(self, llm: LLMEngine, system_prompt: str = DEFAULT_SYSTEM_PROMPT, llm_config: dict | None = None, parent=None):
        self._chatlog = []
        self._llm = llm
        self._llm_config = llm_config if llm_config is not None else {}
        self.parent = parent
        assert parent is not self
        if parent is None:
            self.append({"role": "system", "content": system_prompt})

    def __iter__(self):
        yield from self._chatlog

    def append(self, message: dict[str, str]):
        self._chatlog.append(message)

    async def chat(self, prompt: str, **kwargs):
        _logger.debug(f"Prompting: {prompt}")
        self._chatlog.append({"role": "user", "content": prompt})
        # ValueError on prompt too large.
        message = await self._llm.chat(list(self), **{**self._llm_config, **kwargs})
        _logger.debug(f"Prompt result: {message['content']}")
        self._chatlog.append(message)
        return message['content']

    @contextmanager
    def scope(self):
        chatlog_backup = self._chatlog
        yield
        self._chatlog = chatlog_backup
