from collections import deque
from typing import override, Generator

from model.model import Training, Recipe
from trainer.trainer import Trainer, _logger, next_variation


class RecipeEvaluatorTrainer(Trainer):
    MIN_REVIEWS = 10
    VERY_GOOD_SCORE_THRESHOLD = 4.5
    BAD_SCORE_THRESHOLD = 3.7

    class Variations:
        looks_good = deque([
            "Does this recipe actually look good?",
            "Does this recipe have an appeal?",
            "Is this recipe alluring to you?",
            "Do you perceive this recipe as appetizing?",
            "Does this recipe give off a tasty vibe?",
            "Would you describe this recipe as enticing?",
            "Would you say this recipe looks delicious?",
            "Is this recipe appealing?",
            "Do you find this recipe promising?",
            "Does this recipe seem appetizing?",
            "Does this recipe appear tasty?",
            "Do you think this recipe is enticing?",
            "Does this recipe look delicious?",
            "Does this recipe seem tempting?",
            "How would you rate this recipe?",
            "What rating would you assign to this recipe?",
            "On a scale of one to five, how do you assess this recipe?",
            "Could you share your rating for this recipe?",
            "In your opinion, how does this recipe measure up?",
            "What score would you give to this recipe?",
            "How do you rate the quality of this recipe?",
            "Would you mind providing a rating for this recipe?",
            "On a rating scale, where would you place this recipe?",
            "Can you offer your evaluation of this recipe?",
            "What do you think about assigning a rating to this recipe?"
        ])

        yes_good = deque([
            "It looks very good. I'll give it {}/5\n\n{}",
            "This appears quite appealing. I'd rate it a solid {} out of 5.\n\n{}",
            "It seems delicious. I'm leaning towards a {}/5 score.\n\n{}",
            "From the description alone, it looks very good. A {}/5 seems fitting.\n\n{}",
            "The way it's described makes it sound really appetizing. I'm inclined to rate it {} out of 5.\n\n{}",
            "The recipe paints a tempting picture. I'm thinking a {} out of 5.\n\n{}",
            "The details make it sound quite tasty. I'm going with a {}/5 rating.\n\n{}",
            "It looks incredibly delicious. My rating is settling at {}/5.\n\n{}",
            "Just based on the details provided, it looks very good. I'm leaning towards a {}/5.\n\n{}",
            "The description is making my mouth water. I'm tempted to rate it {} out of 5.\n\n{}",
            "The way the recipe is described is quite enticing. I'm considering a {}/5 rating.\n\n{}",
            "The specifics provided paint a delightful picture. I'm opting for a {}/5 rating.\n\n{}"
        ])

        maybe_good = deque([
            "It can be good. I'll give it {}/5\n\n{}",
            "There's potential here. I'd rate it {} out of 5.\n\n{}",
            "I see promise in this. A solid {}/5 for me.\n\n{}",
            "It has its merits. I'm leaning towards a {} out of 5.\n\n{}",
            "There's something positive about it. I'd assign a {}/5 rating.\n\n{}",
            "I find it decent. A fair evaluation would be {} out of 5.\n\n{}",
            "It's not bad. I'd say it deserves a {}/5.\n\n{}",
            "There's room for improvement, but it's alright—{} out of 5.\n\n{}",
            "I'm optimistic about it. My score would be {}/5.\n\n{}",
            "It's got potential. I'm settling on a {} out of 5.\n\n{}",
            "There's a positive aspect to it. I'd give it a {}/5 rating.\n\n{}"
        ])

        no_good = deque([
            "I'm not convinced. I'd give it a 2.8 out of 5.",
            "Hmm, not really my favorite. I'd rate it 2.8/5.",
            "I'm not feeling it. I'd go with a 2.8 out of 5.",
            "Not my top choice. I'm leaning towards 2.8 out of 5.",
            "I'm not sold on it. A 2.8 out of 5 seems fair.",
            "Not exactly a winner. I'd give it a 2.8/5.",
            "I'm not overly impressed. 2.8 out of 5, I'd say.",
            "I remain unconvinced. My rating would be 2.8 out of 5.",
            "Hmm, it's not quite to my liking. I'd assign it a 2.8/5.",
            "I'm not vibing with it. A 2.8 out of 5 feels right.",
            "It would not be my first choice. I'm inclining towards a 2.8 out of 5.",
            "I'm not entirely convinced. A 2.8 out of 5 seems reasonable.",
            "Not exactly a standout. I'd rate it 2.8/5.",
            "I'm not overly wowed. I'd settle for a 2.8 out of 5."
        ])

        how_to_improve = deque([
             "What changes could enhance this recipe?",
             "In what ways do you think this recipe could be elevated?",
             "Are there any improvements you would suggest for this recipe?",
             "How might we enhance the flavors in this recipe?",
             "Can you think of any adjustments that would make this recipe better?",
             "What improvements do you envision for this particular recipe?",
             "Do you see any room for enhancement in this recipe?",
             "Are there any tweaks or modifications that could enhance this recipe?",
             "In what ways could this recipe be perfected?",
             "Can you suggest any refinements to improve this recipe?"
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
            except Exception:
                _logger.exception(f"Failed to process recipe: {recipe.document.title}")
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

        self._chatlog.append({
            "role": "user",
            "content": f"Starting after the line break are reviews for the recipe.\n\n{recipe.format_reviews()}"
        })

        with self.chat_scope():
            why_good = self._chat(f"Describe why the recipe should get a score of {recipe.review_score}/5. "
                                  "Act like you never saw the reviews. "
                                  "This means you cannot refer to the reviews, reviewers or users in your response.")

        if recipe.review_score > self.VERY_GOOD_SCORE_THRESHOLD:
            yield from self._q_and_q_messages(
                next_variation(self.Variations.looks_good),
                next_variation(self.Variations.yes_good).format(recipe.review_score, why_good).strip())
        else:
            critic = ""
            if self._chat("Is there a concensus amongs the reviews that the recipe could be improved in some way?",
                          grammar=self.GRAMMAR_YES_NO) == "yes":
                critic = self._chat("Write a paragraph that suggest practical improvements to the recipe."
                                    "Write your response using the recipe as the subject of your sentences. "
                                    "Act like you never saw the reviews. "
                                    "This means you cannot refer to the reviews, reviewers or users in your response.")
            if recipe.review_score < self.BAD_SCORE_THRESHOLD:
                yield from self._q_and_q_messages(
                    next_variation(self.Variations.looks_good),
                    next_variation(self.Variations.no_good).format(recipe.review_score, why_good).strip())

                yield from self._q_and_q_messages(next_variation(self.Variations.how_to_improve), critic)
            else:
                yield from self._q_and_q_messages(
                    next_variation(self.Variations.looks_good),
                    next_variation(self.Variations.maybe_good).format(recipe.review_score, why_good).strip())

                yield from self._q_and_q_messages(next_variation(self.Variations.how_to_improve), critic)


if __name__ == '__main__':
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

    training_count = RecipeEvaluatorTrainer(llm, sql, limit=False).start()
    _logger.info(f"Trainer done. It generated {training_count} documents.")
