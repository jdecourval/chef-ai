import abc
import argparse
import json
from typing import override, Generator

import humanize
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
    GRAMMAR_YES_NO = LlamaGrammar.from_string('root ::= "yes" | "no"')

    def __init__(self, llm: Llama, sql: SQLitePipeline):
        self._llm = llm
        self._sql = sql
        self._reset()

    def _reset(self):
        self._chat = [{"role": "system", "content": self.SYSTEM_PROMPT}]

    def _prompt(self, p: str, **kwargs):
        self._chat.append({
            "role": "user",
            "content": p
        })
        try:
            message = llm.create_chat_completion(self._chat, **{**self.CHAT_DEFAULTS, **kwargs})['choices'][0]['message']
            # print(message['content'])
            self._chat.append(message)
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


class RecipeEvaluatorTrainer(Trainer):
    MIN_REVIEWS = 10
    VERY_GOOD_SCORE_THRESHOLD = 4.5
    BAD_SCORE_THRESHOLD = 3.7

    @override
    def __iter__(self) -> Generator[list[dict[str, str]], None, None]:
        # TODO: If too much memory or latency, do it element per element, or async.
        recipes = [Recipe(**i) for i in self._sql.select("SELECT * FROM Recipe")]
        for recipe, document in zip(recipes, self._sql.select(
                f"SELECT * FROM Document WHERE id IN ({",".join('?' * len(recipes))})",
                [j.document for j in recipes])):
            recipe.document = Document(**document)
        for recipe in recipes:
            # TODO: Add marker between conversations
            yield self._process_document(recipe)
            self._reset()

    def _process_document(self, recipe: Recipe) -> list[dict[str, str]] | None:
        if recipe.review_count is None or recipe.review_score is None:
            return None

        doc = recipe.document
        conversation = [{
            "role": "user",
            "content": "Starting after the line break is a RECIPE by a food magazine.\n\n" + doc.text
        }]

        if recipe.review_count > self.MIN_REVIEWS:
            if recipe.review_score > self.VERY_GOOD_SCORE_THRESHOLD:
                conversation += self._q_and_q_messages("Does this recipe actually looks good?",
                                                       f"Yes, very good. I'll give it {recipe.review_score}/5")
            elif recipe.review_score < self.BAD_SCORE_THRESHOLD:
                conversation += self._q_and_q_messages("Does this recipe actually looks good?",
                                                       f"No, not really. I'll give it {recipe.review_score}/5")
            else:
                conversation += self._q_and_q_messages("Does this recipe actually looks good?",
                                                       f"It can be good. I'll give it {recipe.review_score}/5")

        # TODO: Summarize all the reviews, and append to the response.
        return conversation


class RecipeTrainer(Trainer):
    MIN_SCORE = 3.5

    @override
    def __iter__(self) -> Generator[list[dict[str, str]], None, None]:
        # TODO: If too much memory or latency, do it element per element, or async.
        recipes = [Recipe(**i) for i in self._sql.select("SELECT * FROM Recipe")]
        for recipe, document in zip(recipes, self._sql.select(
                f"SELECT * FROM Document WHERE id IN ({",".join('?' * len(recipes))})",
                [j.document for j in recipes])):  # Generators are not support by sqlite
            recipe.document = Document(**document)
        for recipe in recipes:
            for conversation in self._process_document(recipe):
                # TODO: Add marker between conversations
                yield conversation
            self._reset()

    def _process_document(self, recipe: Recipe) -> list[dict[str, str]] | None:
        if recipe.review_score is None or recipe.review_score < self.MIN_SCORE:
            return

        doc = recipe.document
        self._chat.append({
            "role": "user",
            "content": "Starting after the line break is a RECIPE by a food magazine.\n\n" + doc.text
        })

        # TODO: Reset point. Use context manager?

        # Maybe redundant with SummarizingTrainer? Probably different enough.
        # TODO: Finish implementing
        answer = self._prompt(
            "Is there a secret, a key technique, or a special ingredient to this recipe that contributes to its success?")
        if self._prompt("Is that common knowledge?", grammar=self.GRAMMAR_YES_NO) == "yes":
            return
        answer = self._prompt("Anything else?")
        if self._prompt("Is that common knowledge, or obvious to most people?", grammar=self.GRAMMAR_YES_NO) == "yes":
            return

        self._reset()
        self._chat.append({
            "role": "user",
            "content": "Starting after the line break is a RECIPE by a food magazine.\n\n" + doc.text
        })
        title = self._prompt(
            f'The original title of the recipe is "{doc.title}".'
            f' Considering your knowledge of the recipe, can you simplify the title to be a description, as simple as possible, of what is being cooked? Does not use the word "recipe".'
            f' You response must only include the new title.')
        if self._prompt("Is that new title simpler than the original title?", grammar=self.GRAMMAR_YES_NO) == "no":
            title = doc.title  # TODO: Seems useless. Measure.

        conversation = [
            *self._q_and_q_messages(
                f"I'd to cook {title} tonight. Can you propose a recipe?",
                repr(recipe)
            ),
            *self._q_and_q_messages(
                "What are the nutrition facts?",
                recipe.format_nutrition()),
            *self._q_and_q_messages(
                "What sort of cuisine is that?",
                ", ".join(recipe.cuisine[:-1]) + " and " + recipe.cuisine[-1] if len(recipe.cuisine) > 1 else recipe.cuisine[0]
            )]

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

        yield conversation


class SummarizingTrainer(Trainer):
    @override
    def __iter__(self) -> Generator[list[dict[str, str]], None, None]:
        for document in (Document(**i) for i in self._sql.select("SELECT * FROM Document")):
            # TODO: Add marker between conversations
            yield self._process_document(document)
            self._reset()

    def _build_conversation(self, question: str, summary: str) -> list[dict[str, str]]:
        return [{"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": question},
                {"role": "assistant", "content": summary}]

    def _process_document(self, doc: Document) -> list[dict[str, str]] | None:
        self._chat.append({
            "role": "user",
            "content": "Starting after the line break is an ARTICLE by a food magazine.\n\n" + doc.text
        })

        question = self._prompt("What QUESTION could that ARTICLE helps answer? "
                                "Respond only with the QUESTION which must end with a question mark(?).")

        if self._prompt(
                "Does the ARTICLE talks of anecdotes, does it tell a story, or is it about culinary knowledge? Your "
                "answer must be one word: 'anecdotes', 'story' or 'knowledge'.",
                grammar=LlamaGrammar.from_string(
                    'root ::= "anecdotes" | "story" | "knowledge"')) in ["anecdotes", "story"]:
            return None

        worth_it = self._prompt(
            "Would that be useful to train the answer to that QUESTION to an AI specialized in cooking and food in "
            "general? Consider that the AI has a limited memory, therefore, training articles based on anecdotes or "
            "stories must be avoided. Don't try to compromize.",
            grammar=self.GRAMMAR_YES_NO) == "yes"
        if not worth_it:
            return

        self._reset()
        self._chat.append({
            "role": "user",
            "content": "Starting after the line break is an ARTICLE by a food magazine.\n\n" + doc.text
        })
        # TODO: Sometimes, the answer still refers to "the/this article", or "the/this recipe".
        summary = self._prompt(
            f'Answer, in your own words, not the author\'s, the question "{question}" from the ARTICLE\'s content. '
            f'Explain step-by-step, miticulously. Respond only with your long and detailed answer. '
            f'Elaborate in up to 600 words.',
            max_tokens=2000)

        return self._build_conversation(question, summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    args = parser.parse_args()
    llm = Llama(model_path=args.model, n_gpu_layers=99, n_ctx=16*1024, chat_format="chatml", verbose=False)
    sql = SQLitePipeline()

    with open("recipes.jsonl", "w") as file:
        for i in RecipeTrainer(llm, sql):
            print(i)
            json.dump(i, file, ensure_ascii=False)

    with open("reviews.jsonl", "w") as file:
        for i in RecipeEvaluatorTrainer(llm, sql):
            print(i)
            json.dump(i, file, ensure_ascii=False)

    with open("summaries.jsonl", "w") as file:
        for i in SummarizingTrainer(llm, sql):
            print(i)
            json.dump(i, file, ensure_ascii=False)
