import abc
import dataclasses
import logging
import uuid
from abc import ABC
from collections import deque
from contextlib import contextmanager
from typing import Type, AsyncGenerator, TypeVar, override

from ai.engine import LLMEngine
from db.db import SQLitePipeline
from model.model import Document, Recipe, Training
from utils.generator import first

# TODO: Try this model to generate questions: https://huggingface.co/FPHam/Generate_Question_Mistral_7B
# TODO: Generate content from charts/pictures.
# TODO: CoT or similar prompting style. Read Orca paper.
# TODO: Metrics


_logger = logging.getLogger(__name__)


def next_variation(variations: deque):
    item = variations.popleft()
    variations.append(item)
    return item


class CategoryIngredientTrainer:
    # TODO:
    #  1. Generate an ingredient and a category.
    #  2. Search the DB for it. Order by the ratio of that ingredient if possible, filter by score, if there's one, and by category.
    #  3. Build a conversion:
    #     - What would be a good recipe for my X.
    #     - You could try:
    #       - X
    #       - X
    pass


class Trainer(ABC):
    Input = TypeVar("Input")
    SYSTEM_PROMPT = ("You are a helpful assistant. Below is an instruction that describes a task. "
                     "Write a response that appropriately completes the request.")
    _CHAT_DEFAULTS = {"max_tokens": 400, "temperature": 0}
    _LIMIT_QUICK = "ORDER BY RANDOM() LIMIT 50"

    class _ChatScope:
        def __init__(self, llm: LLMEngine, parent=None):
            self._chatlog = []
            self._llm = llm
            self.parent = parent
            assert parent is not self

        def __iter__(self):
            if self.parent is not None:
                yield from self.parent
            yield from self._chatlog

        def append(self, message: dict[str, str]):
            self._chatlog.append(message)

        async def chat(self, prompt: str, **kwargs):
            _logger.debug(f"Prompting: {prompt}")
            self._chatlog.append({"role": "user", "content": prompt})
            # ValueError on prompt too large.
            message = await self._llm.chat(list(self), **{**Trainer._CHAT_DEFAULTS, **kwargs})
            _logger.debug(f"Prompt result: {message['content']}")
            self._chatlog.append(message)
            return message['content']

    def __init__(self, input: Input, llm: LLMEngine, revision: str = "", embed_model=None):
        self._llm = llm
        self.revision = revision
        self.chat = self._ChatScope(self._llm)
        self.chat.append({"role": "system", "content": Trainer.SYSTEM_PROMPT})
        self.grammar_yes_no = self._llm.get_options_grammar(("yes", "no"))
        self.input = input
        self._new_conversation()
        # self.embed_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    @classmethod
    @abc.abstractmethod
    async def document_generator(cls, sql: SQLitePipeline, revision: str, quick=False) -> AsyncGenerator[Input, None]:
        pass

    @classmethod
    @abc.abstractmethod
    def total_document(cls, sql: SQLitePipeline, revision, quick=False) -> int:
        pass

    @abc.abstractmethod
    async def __aiter__(self) -> AsyncGenerator[Training, None]:
        pass

    @contextmanager
    def _chat_scope(self):
        self.chat = Trainer._ChatScope(self._llm, self.chat)
        yield
        self.chat = self.chat.parent

    async def _prompt(self, prompt: str, **kwargs):
        return (await self._llm.chat([
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ], **{**self._CHAT_DEFAULTS, **kwargs}))['content']

    def _q_and_q_training(self, question, answer):
        conversation = (self._training({"role": "user", "content": question}),
                        self._training({"role": "assistant", "content": answer}))
        return conversation

    def _new_conversation(self):
        self.position = -1  # Get incremented on first use.
        self.conversation_id = uuid.uuid4()

    def _training(self, conversation: dict[str, str]) -> Training:
        self.position += 1
        return Training(conversation=self.conversation_id,
                        position=self.position,
                        content=conversation["content"],
                        role=Training.Role[conversation["role"]],
                        trainer=self.__class__.__name__,
                        source=self.input.document if isinstance(self.input, Recipe) else self.input,
                        revision=self.revision
                        )


class RecipeTrainerBase(Trainer, ABC):
    @classmethod
    @override
    def total_document(cls, sql: SQLitePipeline, revision, quick=False) -> int:
        if quick:
            return 50
        return first(sql.select_one_col(
            "SELECT count(1) FROM Recipe "
            "INNER JOIN Document on Document.id = Recipe.document "
            "LEFT JOIN Training ON Document.id=Training.source AND Training.trainer=? AND Training.revision=? "
            "WHERE Training.source IS NULL "
            f"{cls._LIMIT_QUICK if quick else ''} ", (cls.__name__, revision)))

    @classmethod
    @override
    async def document_generator(cls, sql: SQLitePipeline, revision: str,
                                 quick=False) -> AsyncGenerator[Recipe, None]:
        document_fields = [i.name for i in dataclasses.fields(Document)]
        recipe_fields = [i.name for i in dataclasses.fields(Recipe)]
        for recipe_document in sql.select(
                "SELECT Recipe.*, Document.*, Recipe.id as reid FROM Recipe "
                "INNER JOIN Document on Document.id = Recipe.document "
                "LEFT JOIN Training ON Document.id=Training.source AND Training.trainer=? AND Training.revision=? "
                "WHERE Training.source IS NULL "
                f"{cls._LIMIT_QUICK if quick else ''} ", (cls.__name__, revision)):
            document = Document(**{i: j for i, j in recipe_document.items() if i in document_fields})
            recipe = Recipe(**{i: j for i, j in recipe_document.items() if i in recipe_fields})
            recipe.id = recipe_document["reid"]
            recipe.document = document
            yield recipe


def main(trainer_type: Type[Trainer], revision="quick-test", limit=True):
    import logging
    import argparse
    from db.db import SQLitePipeline
    from utils.generator import aenumerate
    from tqdm.asyncio import tqdm
    from ai.engine import LlamaCppPython
    import anyio

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    args = parser.parse_args()

    async def start():
        llm = LlamaCppPython(model=args.model)
        sql = SQLitePipeline()
        total_documents = trainer_type.total_document(sql, revision=revision, quick=limit)
        total_training = 0
        with tqdm(total=trainer_type.total_document(sql, revision=revision, quick=limit)):
            async for document in tqdm(trainer_type.document_generator(sql, quick=limit), total=total_documents):
                async for total_training, training in aenumerate(trainer_type(document, llm, revision=revision)):
                    sql.insert(training)
        return total_training

    training_count = anyio.run(start)
    _logger.info(f"Trainer done. It generated {training_count} documents.")
