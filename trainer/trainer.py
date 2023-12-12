import abc
import argparse
import logging
from contextlib import contextmanager
from typing import override, Generator

import humanize
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
    CHAT_DEFAULTS = {"max_tokens": 200, "temperature": 0}
    GRAMMAR_YES_NO = LlamaGrammar.from_string('root ::= "yes" | "no"', verbose=False)

    def __init__(self, llm: Llama, sql: SQLitePipeline, limit=False):
        self._llm = llm
        self._sql = sql
        self._chatlog = []
        self._limit = "LIMIT 50" if limit else ""
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
        return next(sql.select(f"SELECT count(1) as c FROM {table} {self._limit}"))["c"]

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
        for count, training in enumerate(self):
            self._sql.insert(training)
        return count


class RecipeEvaluatorTrainer(Trainer):
    MIN_REVIEWS = 10
    VERY_GOOD_SCORE_THRESHOLD = 4.5
    BAD_SCORE_THRESHOLD = 3.7

    @override
    def __iter__(self) -> Generator[Training, None, None]:
        for idx, recipe in enumerate(self._all_recipes()):
            try:
                for position, conversation in enumerate(self._process_document(recipe)):
                    yield self._training(conversation=conversation,
                                         conversation_id=idx,
                                         position=position,
                                         source=recipe.document
                                         )
            except Exception as e:
                _logger.exception(f"Failed to process recipe: {recipe.document.title}", e)
            self._reset()

    def _process_document(self, recipe: Recipe) -> Generator[dict[str, str], None, None]:
        if recipe.review_count < self.MIN_REVIEWS:
            return

        intro = {
            "role": "user",
            "content": f"Starting after the line break is a recipe by a food magazine.\n\n{recipe}"
        }

        yield intro
        self._chatlog.append(intro)

        critic = ""
        with self.chat_scope():
            self._chatlog.append({
                "role": "user",
                "content": f"Starting after the line break are reviews for the recipe.\n\n{recipe.format_reviews()}"
            })
            # TODO: This sort of phrasing basically just applies if the recipe is not close to perfect.
            if self._chat("Is there a concensus amongs the reviews that the recipe could be improved in some way?",
                          grammar=self.GRAMMAR_YES_NO) == "no":
                _logger.info("No concensus among the reviews")
                critic = self._chat("Write a paragraph that suggest how to improve this recipe."
                                    "Write your response using the recipe as the subject of your sentences. ")
                # "State the reviewers' conclusions as fact, don't quote them.")

            # Alternative idea:
            # self._chat(f"Describe why the recipe should get a score of {recipe.review_score}/5")

        if recipe.review_score > self.VERY_GOOD_SCORE_THRESHOLD:
            yield from self._q_and_q_messages(
                "Does this recipe actually look good?",
                f"Yes, very good. I'll give it {recipe.review_score}/5\n\n{critic}".strip())
        elif recipe.review_score < self.BAD_SCORE_THRESHOLD:
            yield from self._q_and_q_messages(
                "Does this recipe actually look good?",
                f"No, not really. I'll give it {recipe.review_score}/5\n\n{critic}".strip())
        else:
            yield from self._q_and_q_messages(
                "Does this recipe actually look good?",
                f"It can be good. I'll give it {recipe.review_score}/5\n\n{critic}".strip())


class RecipeTrainer(Trainer):
    MIN_SCORE = 3.5

    @override
    def __iter__(self) -> Generator[Training, None, None]:
        for idx, recipe in enumerate(self._all_recipes()):
            try:
                for position, conversation in enumerate(self._process_document(recipe)):
                    yield self._training(conversation=conversation,
                                         conversation_id=idx,
                                         position=position,
                                         source=recipe.document
                                         )
            except Exception as e:
                _logger.exception(f"Failed to process recipe: {recipe.document.title}", e)
            self._reset()

    def _process_document(self, recipe: Recipe) -> Generator[dict[str, str], None, None]:
        if recipe.review_score is None or recipe.review_score < self.MIN_SCORE:
            return

        doc = recipe.document
        self._chatlog.append({
            "role": "user",
            "content": "Starting after the line break is a RECIPE by a food magazine.\n\n" + doc.text
        })

        secrets = []
        with self.chat_scope():
            # Maybe redundant with SummarizingTrainer? Probably different enough.
            answer = self._chat(
                "Is there a secret, a key technique, or a special ingredient to this recipe that contributes to its success?")
            if self._chat("Is that common knowledge, or obvious to most people?", grammar=self.GRAMMAR_YES_NO) == "no":
                secrets.append(answer)

                for _ in range(3):
                    answer = self._chat("Anything else?")
                    if answer.lower().startswith("no") or self._chat(
                            "Is that common knowledge, or obvious to most people?",
                            grammar=self.GRAMMAR_YES_NO) == "yes":
                        break
                    secrets.append(answer)
                else:
                    _logger.info("Generated too many techniques from the recipe.")

        title = self._chat(
            f'The original title of the recipe is: "{doc.title}". '
            f' What is being cooked in this recipe? '
            f'Your response must be terse, only include the answer to the question and not use the word "recipe" nor any verb.',
            max_tokens=8)
        title = title.strip('"')  # Sometimes the response is quoted.

        question = self._prompt("Can you fix this sentence to be more grammatically correct? "
                                "Your response must only include the fixed sentence.\n\n"
                                f"I'd like to prepare {title} tonight. Can you propose a recipe?")
        yield from self._q_and_q_messages(question, repr(recipe))
        yield from self._q_and_q_messages("What are the nutrition facts?", recipe.format_nutrition())

        if recipe.cuisine:
            yield from self._q_and_q_messages(
                "What sort of cuisine is that?",
                ", ".join(recipe.cuisine[:-1]) + " and " + recipe.cuisine[-1] if len(recipe.cuisine) > 1
                else recipe.cuisine[0]
            )

        if recipe.category:
            yield from self._q_and_q_messages(
                "How would you caracterize this recipe?",
                "As a " + ", ".join(recipe.category[:-1]) + " and " + recipe.category[-1] if len(recipe.category) > 1
                else recipe.category[0])

        if recipe.prep_time or recipe.total_time:
            if recipe.prep_time and recipe.total_time:
                time_answer = (f"This recipe needs {humanize.naturaldelta(recipe.prep_time)} to prepare and "
                               f"{humanize.naturaldelta(recipe.total_time)} in total.")
            elif recipe.prep_time:
                time_answer = f"This recipe needs {humanize.naturaldelta(recipe.prep_time)} to prepare."
            else:
                time_answer = f"This recipe needs {humanize.naturaldelta(recipe.total_time)} in total."
            yield from self._q_and_q_messages(
                "How long does the recipe take?",
                time_answer
            )

        if secrets:
            yield from self._q_and_q_messages(
                "How can I make sure this recipe is a success?",
                self._prompt(
                    "Summarize, simplify, and improve the wording of the following text:\n\n" + "\n".join(secrets))
            )


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
                # TODO: Compare embeddings before adding to the list.
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


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    args = parser.parse_args()
    llm = Llama(model_path=args.model, n_gpu_layers=99, n_ctx=16 * 1024, chat_format="chatml", verbose=False,
                embedding=True)
    # llm.set_cache(LlamaRAMCache(100 * 1024 ** 2))  # This seems to massively increase RAM usage and slow down overall.
    sql = SQLitePipeline()

    for trainer in RecipeEvaluatorTrainer, RecipeTrainer, SummarizingTrainer:
        _logger.info(f"Starting trainer: {trainer.__name__}")
        training_count = trainer(llm, sql, limit=True).start()
        _logger.info(f"Trainer done. It generated {training_count} documents.")
