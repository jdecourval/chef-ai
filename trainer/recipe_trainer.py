from collections import deque
from typing import override, Generator

import humanize

from model.model import Training, Recipe
from trainer.trainer import Trainer, _logger


class RecipeTrainer(Trainer):
    MIN_SCORE = 3.5

    class Variations:
        request_help = deque([
            "I'd like to prepare {} tonight. Can you propose a recipe?",
            "Thinking of making {} for dinner tonight. Any recipe suggestions you can share?",
            "Considering {} for tonight's meal. Any recipe ideas you could recommend?",
            "Contemplating a {} dish for dinner. Mind sharing a recipe suggestion?",
            "In the mood for {} tonight. Any chance you could suggest a recipe?",
            "Craving {} for dinner. Can you recommend a recipe to prepare?",
            "Interested in cooking {} tonight. Any recipe suggestions from your culinary expertise?",
            "Considering a {} dish for tonight. Any go-to recipes you'd recommend trying?",
            "Contemplating {} as the main course. Could you propose a recipe for tonight's dinner?",
            "Feeling like preparing {} tonight. Do you have a recipe you'd suggest for me?"
        ])

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
