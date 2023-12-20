import abc
import logging
from collections import deque
from contextlib import contextmanager
from typing import Generator, Type

from llama_cpp import Llama, LlamaGrammar
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from db.db import SQLitePipeline
from model.model import Document, Recipe, Training

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
    GRAMMAR_YES_NO = LlamaGrammar.from_string('root ::= "yes" | "no"', verbose=False)
    GRAMMAR_LIST = LlamaGrammar.from_string(r'''root ::= item+
item ::= "- " [^\r\n\x0b\x0c\x85\u2028\u2029]+ "\n"''', verbose=False)
    # GRAMMAR_LIST_NUMBERED = LlamaGrammar.from_string(r'''root ::= item+
    # item ::= "\d+\. " [^\r\n\x0b\x0c\x85\u2028\u2029]+ "\n"''', verbose=False)

    def __init__(self, llm: Llama, sql: SQLitePipeline, limit=False):
        self._llm = llm
        self._sql = sql
        self._chatlog = []
        self._limit = "ORDER BY RANDOM() LIMIT 50" if limit else ""
        self.embed_model = SentenceTransformer('thenlper/gte-large')
        # self.embed_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        self._reset()

    @contextmanager
    def chat_scope(self):
        old_size = len(self._chatlog)
        yield
        self._chatlog = self._chatlog[:old_size]

    def _reset(self):
        self._chatlog = [{"role": "system", "content": self.SYSTEM_PROMPT}]

    def _prompt(self, prompt: str, **kwargs):
        return self._llm.create_chat_completion([
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ], **{**self.CHAT_DEFAULTS, **kwargs})['choices'][0]['message']['content']

    def _chat(self, prompt: str, **kwargs):
        _logger.debug(f"Prompting: {prompt}")
        self._chatlog.append({"role": "user", "content": prompt})
        # ValueError on prompt too large.
        message = self._llm.create_chat_completion(self._chatlog, **{**self.CHAT_DEFAULTS, **kwargs})['choices'][0][
            'message']
        _logger.debug(f"Prompt result: {message['content']}")
        self._chatlog.append(message)
        return message['content']

    @staticmethod
    def _q_and_q_messages(question, answer):
        return [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]

    @abc.abstractmethod
    def __iter__(self) -> Generator[Training, None, None]:
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
                        source=source
                        )

    def start(self):
        count = 0
        for count, training in enumerate(self):
            self._sql.insert(training)
        return count


class RecipeTrainerBase(Trainer):
    def __iter__(self) -> Generator[Training, None, None]:
        for idx, recipe in enumerate(self._all_recipes()):
            if next(self._sql.select_one_col("SELECT count(1) FROM Training "
                                             "WHERE source=? AND trainer='RecipeEvaluatorTrainer'", (recipe.document,))):
                _logger.info(f"Skipping over already processed recipe: {recipe}")
                continue
            try:
                for position, conversation in enumerate(self._process_document(recipe)):
                    yield self._training(conversation=conversation,
                                         conversation_id=idx,
                                         position=position,
                                         source=recipe.document
                                         )
            except:
                _logger.exception(f"Failed to process recipe: {recipe}")
            self._reset()

    @abc.abstractmethod
    def _process_document(self, recipe: Recipe) -> Generator[dict[str, str], None, None]:
        pass


def main(trainer: Type[Trainer], limit=False):
    import logging
    import argparse
    from llama_cpp import Llama
    from db.db import SQLitePipeline

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    args = parser.parse_args()
    llm = Llama(model_path=args.model, n_gpu_layers=99, n_ctx=16 * 1024, chat_format="chatml", verbose=False,
                embedding=True)
    # llm.set_cache(LlamaRAMCache(100 * 1024 ** 2))  # This seems to massively increase RAM usage and slow down overall.
    sql = SQLitePipeline()

    training_count = trainer(llm, sql, limit=limit).start()
    _logger.info(f"Trainer done. It generated {training_count} documents.")
