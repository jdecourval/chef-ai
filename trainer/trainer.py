import abc
import logging
from collections import deque
from contextlib import contextmanager
from typing import Generator, Type, AsyncGenerator

import anyio
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from ai.engine import LLMEngine, LlamaCppPython
from db.db import SQLitePipeline
from model.model import Document, Recipe, Training
from utils.aenumerate import aenumerate

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


class Trainer:
    SYSTEM_PROMPT = ("You are a helpful assistant. Below is an instruction that describes a task. "
                     "Write a response that appropriately completes the request.")
    CHAT_DEFAULTS = {"max_tokens": 400, "temperature": 0}

    class ChatScope:
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
            message = await self._llm.chat(list(self), **{**Trainer.CHAT_DEFAULTS, **kwargs})
            _logger.debug(f"Prompt result: {message['content']}")
            self._chatlog.append(message)
            return message['content']

    def __init__(self, llm: LLMEngine, sql: SQLitePipeline, revision: str = None, limit=False):
        self._llm = llm
        self._sql = sql
        self._limit = "ORDER BY RANDOM() LIMIT 50" if limit else ""
        self.revision = llm.model_name() if revision is None else revision
        self.embed_model = SentenceTransformer('thenlper/gte-large')
        self.chat = self.ChatScope(self._llm)
        self.chat.append({"role": "system", "content": Trainer.SYSTEM_PROMPT})
        self.grammar_yes_no = self._llm.get_options_grammar(["yes", "no"])
        # self.embed_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    @contextmanager
    def chat_scope(self):
        self.chat = Trainer.ChatScope(self._llm, self.chat)
        yield
        self.chat = self.chat.parent

    async def _prompt(self, prompt: str, **kwargs):
        return (await self._llm.chat([
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ], **{**self.CHAT_DEFAULTS, **kwargs}))['content']

    @staticmethod
    def _q_and_q_messages(question, answer):
        return [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]

    @abc.abstractmethod
    async def __aiter__(self) -> AsyncGenerator[Training, None]:
        pass

    def _count_table(self, table: str) -> int:
        return next(self._sql.select_one_col(f"SELECT count(1) FROM {table}"))

    def _all_recipes(self) -> Generator[Recipe, None, None]:
        # A join would be much faster, but good enough for now.
        recipes = [Recipe(**i) for i in self._sql.select(f"SELECT * FROM Recipe {self._limit}")]
        for recipe, document in zip(
                tqdm(recipes),
                self._sql.select("SELECT * FROM Document "
                                 f"WHERE id IN ({",".join('?' * len(recipes))})", [i.document for i in recipes])):
            recipe.document = Document(**document)
            yield recipe

    def _all_documents(self) -> Generator[Document, None, None]:
        for document in tqdm((Document(**i)
                              for i in self._sql.select(f"SELECT * FROM Document {self._limit}")),
                             total=self._count_table("Document")):
            yield document

    def _insert_training(self, training: Training):
        self._sql.insert(training)

    def _training(self, conversation: dict[str, str], conversation_id: int, position: int,
                  source: Document) -> Training:
        embedding = self.embed_model.encode(conversation["content"], show_progress_bar=False, normalize_embeddings=True)
        return Training(conversation=conversation_id,
                        position=position,
                        content=conversation["content"],
                        role=Training.Role[conversation["role"]],
                        embedding=embedding,
                        trainer=self.__class__.__name__,
                        source=source,
                        revision=self.revision
                        )

    async def start(self):
        count = 0
        async for count, training in aenumerate(self):
            # TODO: Transaction per document
            self._sql.insert(training)
        return count


class RecipeTrainerBase(Trainer):
    async def __aiter__(self) -> AsyncGenerator[Training, None]:
        last_index = next(self._sql.select_one_col(
            f"SELECT coalesce(MAX(conversation), 0) FROM Training WHERE trainer='{self.__class__.__name__}'"))
        for idx, recipe in enumerate(self._all_recipes(), start=last_index + 1):
            if next(self._sql.select_one_col("SELECT count(1) FROM Training "
                                             f"WHERE source=? AND trainer='{self.__class__.__name__}'",
                                             (recipe.document,))):
                _logger.info(f"Skipping over already processed recipe: {recipe.document}")
                continue
            try:
                with self.chat_scope():
                    async for position, conversation in aenumerate(self._process_document(recipe)):
                        yield self._training(conversation=conversation,
                                             conversation_id=idx,
                                             position=position,
                                             source=recipe.document
                                             )
            except:
                _logger.exception(f"Failed to process recipe: {recipe.document}")

    @abc.abstractmethod
    async def _process_document(self, recipe: Recipe) -> Generator[dict[str, str], None, None]:
        pass


def main(trainer: Type[Trainer], revision=None, limit=False):
    import logging
    import argparse
    from db.db import SQLitePipeline

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    args = parser.parse_args()

    async def start():
        llm = LlamaCppPython(model=args.model)
        # llm.set_cache(LlamaRAMCache(100 * 1024 ** 2))  # This seems to massively increase RAM usage and slow down overall.
        sql = SQLitePipeline()
        await trainer(llm, sql, revision=revision, limit=limit).start()

    training_count = anyio.run(start)
    _logger.info(f"Trainer done. It generated {training_count} documents.")
