import abc
import logging
from collections import deque
from contextlib import contextmanager
from typing import override, Generator

import numpy as np
from llama_cpp import Llama, LlamaGrammar
from tqdm import tqdm

from db.db import SQLitePipeline
from model.model import Document, Recipe, Training

# TODO: Use multiple similar variations of each prompt in the training set.
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

    def __init__(self, llm: Llama, sql: SQLitePipeline, limit=False):
        self._llm = llm
        self._sql = sql
        self._chatlog = []
        self._limit = "ORDER BY RANDOM() LIMIT 50" if limit else ""
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
        return next(self._sql.select(f"SELECT count(1) as c FROM {table} {self._limit}"))["c"]

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
        embedding = self._llm.embed(conversation["content"])
        return Training(conversation=conversation_id,
                        position=position,
                        content=conversation["content"],
                        role=Training.Role[conversation["role"]],
                        embedding=np.array(embedding),
                        trainer=self.__class__.__name__,
                        source=source
                        )

    def start(self):
        count = 0
        for count, training in enumerate(self):
            self._sql.insert(training)
        return count


class SummarizingTrainer(Trainer):
    @override
    def __iter__(self) -> Generator[Training, None, None]:
        for idx, document in enumerate(self._all_documents()):
            try:
                for position, conversation in enumerate(self._process_document(document)):
                    yield self._training(conversation=conversation,
                                         conversation_id=idx,
                                         position=position,
                                         source=document
                                         )
            except Exception as e:
                _logger.exception(f"Failed to process recipe: {document.title}", e)
            self._reset()

    def _process_document(self, doc: Document) -> Generator[dict[str, str], None, None]:
        self._chatlog.append({
            "role": "user",  # Using system breaks the next prompt.
            "content": "Starting after the line break is an ARTICLE by a food magazine.\n\n" + doc.text
        })

        with self.chat_scope():
            if self._chat(
                    "Does the ARTICLE talks of anecdotes, does it tell a story, or is it about culinary knowledge? Your "
                    "answer must be one word: 'anecdotes', 'story' or 'knowledge'.",
                    grammar=LlamaGrammar.from_string('root ::= "anecdotes" | "story" | "knowledge"', verbose=False)
            ) in ["anecdotes", "story"]:
                return

        # TODO: Alternative idea: Ask all questions at the same time.
        questions = []
        with self.chat_scope():
            question = self._chat("Is there a cooking related QUESTION that the content of this article would answer? "
                                  'Respond only with the QUESTION which must end with a question mark(?). '
                                  'Avoid using the words: "article" and "author".')

            while not question.lower().startswith("no") and self._chat(
                    "Would that be useful to train the answer to this QUESTION to an AI specialized in cooking and food in "
                    "general? Consider that the AI has a limited memory, therefore, training articles based on anecdotes or "
                    "stories must be avoided. Don't try to compromize.",
                    grammar=self.GRAMMAR_YES_NO) == "yes":
                # TODO: Move this to post-processing
                # TODO: Try asking the model to rephrase the sentence instead
                # if not re.search("article|author", question, flags=re.IGNORECASE):
                questions.append(question)
                if len(questions) == 4:
                    _logger.info("Generated too many questions")
                    break
                question = self._chat(
                    'Is there another cooking related QUESTION that the content of this article would answer? '
                    'Respond only with the QUESTION which must end with a question mark(?), or with "no" if you can\'t think of any new original question. '
                    'Respond with "no" if your question is similar to a previous one. '
                    'Your response MUST NOT use the words: "article" nor "author".')

        for question in questions:
            with self.chat_scope():
                summary = self._chat(
                    f'Answer, in your own words, not the author\'s, the question "{question}" from the ARTICLE\'s content. '
                    f'Explain step-by-step, miticulously. Respond only with your long and detailed answer. '
                    f'Elaborate in up to 600 words. Avoid using the words: "article" and "author".',
                    max_tokens=2000)

                yield from self._q_and_q_messages(question, summary)

