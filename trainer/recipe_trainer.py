import logging
from collections import deque
from typing import override, Generator

import humanize

from model.model import Recipe
from trainer.trainer import next_variation, main, RecipeTrainerBase

_logger = logging.getLogger(__name__)


class RecipeTrainer(RecipeTrainerBase):
    MIN_SCORE = 4.0

    class Variations:
        propose_recipe = deque([
            ("I'd like to prepare [] tonight.", " Can you propose a recipe?"),
            ("Thinking of making [] for dinner tonight.", " Any recipe suggestions you can share?"),
            ("Considering [] for tonight's meal.", " Any recipe ideas you could recommend?"),
            ("Contemplating a [] dish for dinner.", " Mind sharing a recipe suggestion?"),
            ("In the mood for [] tonight.", " Any chance you could suggest a recipe?"),
            ("Craving [] for dinner.", " Can you recommend a recipe to prepare?"),
            ("Interested in cooking [] tonight.", " Any recipe suggestions from your culinary expertise?"),
            ("Considering a [] dish for tonight.", " Any go-to recipes you'd recommend trying?"),
            ("Contemplating [] as the main course.", " Could you propose a recipe for tonight's dinner?"),
            ("Feeling like preparing [] tonight.", " Do you have a recipe you'd suggest for me?")
        ])

        give_nutrition = deque([
            "What are the nutrition facts?",
            "Could you provide me with the nutritional information?",
            "I'm interested in knowing the nutritional details. Where can I find them?",
            "Can you share the nutritional facts for this recipe?",
            "I'd like to learn more about the nutritional content.",
            "What are the nutritional values for this dish?",
            "Do you have the nutritional information available for this recipe?",
            "I'm looking for details on the nutritional content. Can you help me find them?",
            "What's the nutritional breakdown for this recipe?"
        ])

        which_cuisine = deque([
            "What sort of cuisine is that?",
            "Can you describe the type of cuisine represented in this recipe?",
            "I'm curious about the culinary style of this dish. Can you elaborate?",
            "Could you provide more information about the kind of cuisine featured in this recipe?",
            "I'm interested in knowing the culinary tradition behind this particular dish. What can you tell me?",
            "What type of cuisine does this recipe belong to?",
            "Can you paint a picture of the culinary style reflected in this recipe?",
            "I'd like to learn more about the culinary origins of this dish. What can you share?",
            "Could you give me some insights into the kind of cuisine this recipe falls under?",
            "What culinary influences can be identified in this particular recipe?",
            "I'm trying to grasp the essence of the cuisine showcased in this recipe. Any details you can provide?"
        ])

        which_category = deque([
            "How would you characterize this recipe?",
            "How might you label this recipe?",
            "In what categories would you place this recipe?",
            "How do you define the nature of this recipe?",
            "What classification would you assign to this recipe?",
            "How do you categorize this particular recipe?",
            "In what culinary context would you position this recipe?",
            "How would you typify this recipe?",
            "What characterization would you give to this recipe?",
            "In what culinary genre would you place this recipe?",
            "How do you profile this recipe?",
            "How would you identify the type of this recipe?",
            "In what gastronomic class would you locate this recipe?",
            "How might you pigeonhole this recipe?",
            "What descriptive category would you assign to this recipe?",
            "How do you distinguish the classification of this recipe?"
        ])

        cuisine_answer = deque([
            "I would classify as "
        ])

        category_answer = deque([
            "I would say it's a "
        ])

        how_long_question = deque([
            "What's the time commitment for this recipe?",
            "Can you tell me the time required for this recipe?",
            "How much time does it take to make this recipe?",
            "I'm curious about the time investment for this recipe. What is it?",
            "Could you give me an idea of the time needed for this recipe?",
            "What's the time duration for making this recipe?",
            "Do you know how long it takes to prepare this recipe?",
            "I'm wondering about the time it takes for this recipe. Can you share that information?",
            "Could you inform me about the time required for this recipe?",
            "What's the time commitment involved in making this recipe?"
        ])

        prep_time = deque([
            "The preparation time for this recipe is {}.",
            "To prepare this recipe, you'll need {}.",
            "The recipe calls for {} of preparation.",
            "You'll be spending {} on preparation.",
            "For preparation, allocate {}.",
            "Expect to spend {} on preparation.",
            "The preparation time for this recipe is {}.",
            "You'll need {} for preparation.",
            "The recipe involves {} of preparation.",
            "Plan for {} of preparation."
        ])

        total_time = deque([
            "The total time required for this recipe is {}.",
            "Sure, this recipe takes about {} to complete.",
            "You'll need approximately {} to prepare this recipe.",
            "Expect to spend around {} on this recipe.",
            "To make this recipe, plan on dedicating {} of your time.",
            "The estimated time for completing this recipe is {}.",
            "Certainly, you should set aside {} for this recipe.",
            "You'll be looking at a total time of {} for this recipe.",
            "Sure, you'll be spending around {} on this recipe.",
            "Plan on spending {} to make this recipe."
        ])

        prep_and_total_time = deque([
            "The preparation time for this recipe is {}, and the total time is {}.",
            "To prepare this recipe, you'll need {}, and the total time is {}.",
            "The recipe calls for {} of preparation and a total of {}.",
            "You'll be spending {} on preparation, and the total time for this recipe is {}.",
            "For preparation, allocate {}, and the total time required is {}.",
            "Expect to spend {} on preparation, with a total time of {} for this recipe.",
            "The preparation time for this recipe is {}, and the overall time investment is {}.",
            "You'll need {} for preparation, and the total time for this recipe is {}.",
            "The recipe involves {} of preparation and a total time of {}.",
            "Plan for {} of preparation, and the total time needed is {}."
        ])

        how_to_suceed = deque([
            "How can I make sure this recipe is a success?",
            "What steps should I take to ensure the success of this recipe?",
            "How do I guarantee the success of this recipe?",
            "In making this recipe, what can I do to make sure it turns out well?",
            "What measures should I take to make sure this recipe is a success?",
            "What can I do to make sure this recipe turns out perfectly?",
            "How can I guarantee the success of this recipe from the book I'm following?",
            "What steps should I follow to make sure this recipe is a hit?",
            "How can I ensure the success of this recipe I'm trying out?",
            "What can I do to make sure this recipe turns out delicious?",
            "How do I make sure this recipe I found in the book is successful?"
        ])

    @override
    def _process_document(self, recipe: Recipe) -> Generator[dict[str, str], None, None]:
        if recipe.review_score is None or recipe.review_score < self.MIN_SCORE:
            return

        doc = recipe.document
        self._chatlog.append({
            "role": "user",
            "content": "Starting after the line break is a RECIPE by a food magazine.\n\n" + doc.text
        })

        # TODO: Optimize this
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

        title_variation = next_variation(self.Variations.propose_recipe)
        title = self._chat(
            f'Optional context: The original title of the recipe is: "{doc.title}". '
            f'Answer the question: What is being cooked in this recipe? '
            f'Your answer must complete the sentence: "{title_variation[0]}". '
            "The sentence includes a template marker []. You must substitute the template marker by your answer. "
            "Respond with the completed sentence only.",
            max_tokens=20) + title_variation[1]
        yield from self._q_and_q_messages(title, repr(recipe))
        yield from self._q_and_q_messages(next_variation(self.Variations.give_nutrition), recipe.format_nutrition())

        if recipe.cuisine:
            yield from self._q_and_q_messages(
                next_variation(self.Variations.which_cuisine),
                # TODO: prompt: Write a sentence that briefly answers the question considering the answer is {}.
                next_variation(self.Variations.cuisine_answer) + (", ".join(recipe.cuisine[:-1]) + " and " +
                                                                  recipe.cuisine[-1] if len(recipe.cuisine) > 1
                                                                  else recipe.cuisine[0])
            )

        if recipe.category:
            yield from self._q_and_q_messages(
                next_variation(self.Variations.which_category),
                # TODO: prompt: Write a sentence that briefly answers the question considering the answer is {}.
                next_variation(self.Variations.category_answer) + (", ".join(recipe.category[:-1]) + " and " +
                                                                   recipe.category[-1] if len(recipe.category) > 1
                                                                   else recipe.category[0]))

        if recipe.prep_time or recipe.total_time:
            if recipe.prep_time and recipe.total_time:
                time_answer = next_variation(self.Variations.prep_and_total_time).format(
                    humanize.naturaldelta(recipe.prep_time),
                    humanize.naturaldelta(recipe.total_time))
            elif recipe.prep_time:
                time_answer = next_variation(self.Variations.prep_time).format(humanize.naturaldelta(recipe.prep_time))
            else:
                time_answer = next_variation(self.Variations.total_time).format(
                    humanize.naturaldelta(recipe.total_time))

            yield from self._q_and_q_messages(next_variation(self.Variations.how_long_question), time_answer)

        if secrets:
            yield from self._q_and_q_messages(next_variation(self.Variations.how_to_suceed), self._prompt(
                "Summarize, simplify, and improve the wording of the following text:\n\n" + "\n".join(secrets)))


if __name__ == '__main__':
    main(RecipeTrainer)
