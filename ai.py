import abc
import argparse
import json
import re
from contextlib import contextmanager
from typing import override, Generator

import humanize
import tqdm
from llama_cpp import Llama, LlamaGrammar

from db import SQLitePipeline
from model import Document, Recipe


# TODO: Use embeddings to remove duplicated questions (and answers?).
# TODO: Use multiple similar variations of each prompt in the training set.
# TODO: Try this model to generate questions: https://huggingface.co/FPHam/Generate_Question_Mistral_7B
# TODO: Generate content from charts/pictures.
# TODO: CoT or similar prompting style. Read Orca paper.
# TODO: exllamav2
# TODO: Samplers
# TODO: CFG

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
    SYSTEM_PROMPT = ("You are an helpful assistant. Below is an instruction that describes a task. "
                     "Write a response that appropriately completes the request.")
    CHAT_DEFAULTS = {"max_tokens": 200, "temperature": 0}
    GRAMMAR_YES_NO = LlamaGrammar.from_string('root ::= "yes" | "no"', verbose=False)

    def __init__(self, llm: Llama, sql: SQLitePipeline):
        self._llm = llm
        self._sql = sql
        self._chatlog = []
        self._reset()

    @contextmanager
    def chat_scope(self):
        backup = self._chatlog.copy()
        yield
        self._chatlog = backup  # TODO: Use a stack and keep track of the old height.

    def _reset(self):
        self._chatlog = [{"role": "system", "content": self.SYSTEM_PROMPT}]

    def _prompt(self, prompt: str, **kwargs):
        return self._llm.create_chat_completion([
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ], **{**self.CHAT_DEFAULTS, **kwargs})['choices'][0]['message']['content']

    def _chat(self, prompt: str, **kwargs):
        self._chatlog.append({"role": "user", "content": prompt})
        try:
            message = llm.create_chat_completion(self._chatlog, **{**self.CHAT_DEFAULTS, **kwargs})['choices'][0][
                'message']
            # print(message['content'])
            self._chatlog.append(message)
            return message['content']
        except ValueError:
            # Prompt is too large
            print("Too large prompt")

    @staticmethod
    def _q_and_q_messages(question, answer):
        return [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]

    @abc.abstractmethod
    def __iter__(self) -> Generator[list[dict[str, str]], None, None]:
        pass

    def _all_recipes(self):
        # A join would be much faster, but good enough for now.
        recipes = [Recipe(**i) for i in self._sql.select("SELECT * FROM Recipe")]
        for recipe, document in zip(recipes, self._sql.select(
                f"SELECT * FROM Document WHERE id IN ({",".join('?' * len(recipes))})",
                [j.document for j in recipes])):
            recipe.document = Document(**document)
            yield recipe


class RecipeEvaluatorTrainer(Trainer):
    MIN_REVIEWS = 10
    VERY_GOOD_SCORE_THRESHOLD = 4.5
    BAD_SCORE_THRESHOLD = 3.7

    @override
    def __iter__(self) -> Generator[list[dict[str, str]], None, None]:
        for recipe in self._all_recipes():
            for conversation in self._process_document(recipe):
                yield conversation
            self._reset()

    def _process_document(self, recipe: Recipe) -> Generator[list[dict[str, str]], None, None]:
        if recipe.review_count < self.MIN_REVIEWS:
            return

        conversation = [{
            "role": "user",  # TODO: Try using system here.
            "content": f"Starting after the line break is a recipe by a food magazine.\n\n{recipe}"
        }]
        self._chatlog += conversation

        critic = ""
        with self.chat_scope():
            self._chatlog.append({
                "role": "user",
                "content": f"Starting after the line break are reviews for the recipe.\n\n{recipe.format_reviews()}"
            })
            # TODO: This sort of phrasing basically just applies if the recipe is not close to perfect.
            if self._chat("Is there a concensus amongs the reviews that the recipe could be improved in some way?",
                          grammar=self.GRAMMAR_YES_NO) == "no":
                print("no concensus among the reviews")
                critic = self._chat("Write a paragraph that suggest how to improve this recipe."
                                    "Write your response using the recipe as the subject of your sentences. ")
                # "State the reviewers' conclusions as fact, don't quote them.")

            # Alternative idea:
            # self._chat(f"Describe why the recipe should get a score of {recipe.review_score}/5")

        if recipe.review_score > self.VERY_GOOD_SCORE_THRESHOLD:
            conversation += self._q_and_q_messages("Does this recipe actually look good?",
                                                   f"Yes, very good. I'll give it {recipe.review_score}/5\n\n" + critic)
        elif recipe.review_score < self.BAD_SCORE_THRESHOLD:
            conversation += self._q_and_q_messages("Does this recipe actually look good?",
                                                   f"No, not really. I'll give it {recipe.review_score}/5\n\n" + critic)
        else:
            conversation += self._q_and_q_messages("Does this recipe actually look good?",
                                                   f"It can be good. I'll give it {recipe.review_score}/5\n\n" + critic)

        yield conversation


class RecipeTrainer(Trainer):
    MIN_SCORE = 3.5

    @override
    def __iter__(self) -> Generator[list[dict[str, str]], None, None]:
        for recipe in self._all_recipes():
            for conversation in self._process_document(recipe):
                yield conversation
            self._reset()

    def _process_document(self, recipe: Recipe) -> Generator[list[dict[str, str]], None, None]:
        if recipe.review_score is None or recipe.review_score < self.MIN_SCORE:
            return

        doc = recipe.document
        self._chatlog.append({
            "role": "system",
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
                    print("Tried four times.")

        title = self._chat(
            f'The original title of the recipe is: "{doc.title}". '
            f' What is being cooked in this recipe? '
            f'Your response must be terse, only include the answer to the question and not use the word "recipe" nor any verb.',
            max_tokens=8)
        title = title.strip('"')  # Sometimes the response is quoted.
        if self._chat(f'Is "{title}" a simpler title simpler than "{doc.title}"?', grammar=self.GRAMMAR_YES_NO) == "no":
            title = doc.title  # TODO: Seems useless. Measure.

        question = self._prompt("Can you fix this sentence to be more grammatically correct? "
                                "Your response must only included the fixed sentence.\n\n"
                                f"I'd like to prepare {title} tonight. Can you propose a recipe?")
        conversation = [
            *self._q_and_q_messages(question, repr(recipe)),
            *self._q_and_q_messages("What are the nutrition facts?", recipe.format_nutrition())
        ]

        if recipe.cuisine:
            conversation += self._q_and_q_messages(
                "What sort of cuisine is that?",
                ", ".join(recipe.cuisine[:-1]) + " and " + recipe.cuisine[-1] if len(recipe.cuisine) > 1
                else recipe.cuisine[0]
            )

        if recipe.category:
            conversation += self._q_and_q_messages(
                "How would you caracterize this recipe?",
                "As a " + ", ".join(recipe.category[:-1]) + " and " + recipe.category[-1] if len(recipe.category) > 1
                else recipe.category[0])

        if recipe.prep_time or recipe.total_time:
            if recipe.prep_time and recipe.total_time:
                time_answer = f"This recipe needs {humanize.naturaldelta(recipe.prep_time)} to prepare and {humanize.naturaldelta(recipe.total_time)} in total."
            elif recipe.prep_time:
                time_answer = f"This recipe needs {humanize.naturaldelta(recipe.prep_time)} to prepare."
            else:
                time_answer = f"This recipe needs {humanize.naturaldelta(recipe.total_time)} in total."
            conversation += self._q_and_q_messages(
                "How long does the recipe take?",
                time_answer
            )

        if secrets:
            conversation += self._q_and_q_messages(
                "How can I make sure this recipe is a success?",
                self._prompt(
                    "Summarize, simplify, and improve the wording of the following text:\n\n" + "\n".join(secrets))
            )

        yield conversation


class SummarizingTrainer(Trainer):
    @override
    def __iter__(self) -> Generator[list[dict[str, str]], None, None]:
        for document in (Document(**i) for i in self._sql.select("SELECT * FROM Document")):
            print(document.title)  # TODO: Remove
            for conversation in self._process_document(document):
                yield conversation
            self._reset()

    def _build_conversation(self, question: str, summary: str) -> list[dict[str, str]]:
        return [{"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": question},
                {"role": "assistant", "content": summary}]

    def _process_document(self, doc: Document) -> Generator[list[dict[str, str]], None, None]:
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
            question = self._chat("Is there a cooking related QUESTION that this article would answer? "
                                  'Respond only with the QUESTION which must end with a question mark(?). '
                                  'Avoid using the words: "article" and "author".')

            while not question.lower().startswith("no") and self._chat(
                    "Would that be useful to train the answer to this QUESTION to an AI specialized in cooking and food in "
                    "general? Consider that the AI has a limited memory, therefore, training articles based on anecdotes or "
                    "stories must be avoided. Don't try to compromize.",
                    grammar=self.GRAMMAR_YES_NO) == "yes":
                # TODO: Compare embeddings before adding to the list.
                if not re.search("article|author", question, flags=re.IGNORECASE):
                    questions.append(question)
                if len(questions) == 4:
                    print("Too many questions")
                    break
                question = self._chat(
                    'Is there another cooking related QUESTION that this article would answer? '
                    'Respond only with the QUESTION which must end with a question mark(?), or with "no" if you can\'t think of any new original question. '
                    'Respond with "no" if your question is similar to a previous one. '
                    'Your response MUST NOT use the words: "article" nor "author"')

        for question in questions:
            with self.chat_scope():
                summary = self._chat(
                    f'Answer, in your own words, not the author\'s, the question "{question}" from the ARTICLE\'s content. '
                    f'Explain step-by-step, miticulously. Respond only with your long and detailed answer. '
                    f'Elaborate in up to 600 words. Avoid using the words: "article" and "author".',
                    max_tokens=2000)

                yield self._build_conversation(question, summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    args = parser.parse_args()
    llm = Llama(model_path=args.model, n_gpu_layers=99, n_ctx=16 * 1024, chat_format="chatml", verbose=False)
    sql = SQLitePipeline()

    recipe_count = next(sql.select("SELECT count(1) as c FROM Recipe"))["c"]
    document_count = next(sql.select("SELECT count(1) as c FROM Document"))["c"]

    with open("recipes.jsonl", "w") as file:
        for training in tqdm.tqdm(RecipeTrainer(llm, sql), total=recipe_count):  # This assumption may not remain true.
            file.write("<s>")
            json.dump(training, file, ensure_ascii=False)

    with open("reviews.jsonl", "w") as file:
        for training in tqdm.tqdm(RecipeEvaluatorTrainer(llm, sql), total=recipe_count):
            file.write("<s>")
            json.dump(training, file, ensure_ascii=False)

    with open("summaries.jsonl", "w") as file:
        for training in tqdm.tqdm(SummarizingTrainer(llm, sql), total=document_count):
            file.write("<s>")
            json.dump(training, file, ensure_ascii=False)
