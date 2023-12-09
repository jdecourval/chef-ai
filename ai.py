import abc
import argparse

import humanize
from llama_cpp import Llama, LlamaGrammar

from model import Document


# TODO: Use multiple similar variations of each prompt in the training set.
# TODO: Try this model to generate questions: https://huggingface.co/FPHam/Generate_Question_Mistral_7B
# TODO: Generate content from charts/pictures.
# TODO: CoT or similar prompting style.
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

    def __init__(self, llm):
        self._llm = llm
        self.reset()

    def reset(self):
        self._chat = [{"role": "system", "content": self.SYSTEM_PROMPT}]

    def _prompt(self, p: str, **kwargs):
        self._chat.append({
            "role": "user",
            "content": p
        })
        message = llm.create_chat_completion(self._chat, **{**self.CHAT_DEFAULTS, **kwargs})['choices'][0]['message']
        print(message['content'])
        self._chat.append(message)
        return message['content']

    @staticmethod
    def _q_and_q_messages(question, answer):
        return [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]

    @abc.abstractmethod
    def process_document(self, doc: Document) -> list[dict[str, str]] | None:
        pass


class RecipeTrainer(Trainer):
    MIN_SCORE = 3.5
    MIN_REVIEWS = 10
    VERY_GOOD_SCORE_THRESHOLD = 4.5
    BAD_SCORE_THRESHOLD = 3.7

    def process_document(self, doc: Document) -> list[dict[str, str]] | None:
        if doc.recipe is None:
            return None

        conversation = []

        self._chat.append({
            "role": "user",
            "content": "Starting after the line break is a RECIPE by a food magazine.\n\n" + doc.text
        })

        if doc.recipe.review_count > self.MIN_REVIEWS:
            if doc.recipe.review_score > self.VERY_GOOD_SCORE_THRESHOLD:
                conversation += self._q_and_q_messages("Does this recipe actually looks good?",
                                                       f"Yes, very good. I'll give it {doc.recipe.review_score}/5")
            elif doc.recipe.review_score < self.BAD_SCORE_THRESHOLD:
                conversation += self._q_and_q_messages("Does this recipe actually looks good?",
                                                       f"No, not really. I'll give it {doc.recipe.review_score}/5")

        # TODO: Summarize all the reviews, and append to the response.

        if doc.recipe.review_score and doc.recipe.review_score < self.MIN_SCORE:
            return None

        # TODO: Reset point. Use context manager?

        # Maybe redundant with SummarizingTrainer? Probably different enough.
        answer = self._prompt(
            "Is there a secret, a key technique, or a special ingredient to this recipe that contributes to its success?")
        if self._prompt("Is that common knowledge?", grammar=self.GRAMMAR_YES_NO) == "yes":
            return
        answer = self._prompt("Anything else?")
        if self._prompt("Is that common knowledge, or obvious to most people?", grammar=self.GRAMMAR_YES_NO) == "yes":
            return

        self.reset()
        self._chat.append({
            "role": "user",
            "content": "Starting after the line break is a RECIPE by a food magazine.\n\n" + doc.text
        })
        title = self._prompt(
            'The original title of the recipe is "{}". Can you make a more descriptive title? Does not use the word "recipe". You response must only include the new title.')
        if self._prompt("Is that an improvement on the original title?", grammar=self.GRAMMAR_YES_NO) == "no":
            title = doc.title
        self.reset()

        # Probably no real reason to it now, as opposed to generate the answer now, as opposed to at infereence.
        # Except maybe if the training data are generated with a much better model?
        # self._chat.append({
        #     "role": "user",
        #     "content": "Starting after the line break is a RECIPE.\n\n" + repr(doc.recipe)
        # })
        # self._prompt("Is this recipe healthy")

        conversation = [
            *self._q_and_q_messages(
                f"I'd to cook {title} tonight. Can you propose a recipe?",
                repr(doc.recipe)
            ),
            *self._q_and_q_messages(
                "What are the nutrition facts?",
                repr(doc.recipe.format_nutrition())),
            *self._q_and_q_messages(
                "What sort of cuisine is that?",
                repr(doc.recipe.cuisine)
            )]

        if doc.recipe.prep_time or doc.recipe.total_time:
            if doc.recipe.prep_time and doc.recipe.total_time:
                time_answer = f"This recipe needs {humanize.naturaldelta(doc.recipe.prep_time)} to prepare and {humanize.naturaldelta(doc.recipe.total_time)} in total."
            elif doc.recipe.prep_time:
                time_answer = f"This recipe needs {humanize.naturaldelta(doc.recipe.prep_time)} to prepare."
            else:
                time_answer = f"This recipe needs {humanize.naturaldelta(doc.recipe.total_time)} in total."
            conversation += self._q_and_q_messages(
                "How long does the recipe take?",
                time_answer
            )

        return conversation


class SummarizingTrainer(Trainer):

    def _build_conversation(self, question: str, summary: str) -> list[dict[str, str]]:
        return [{"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": question},
                {"role": "system", "content": summary}]

    def process_document(self, doc: Document) -> list[dict[str, str]] | None:
        self._chat.append({
            "role": "user",
            "content": "Starting after the line break is an ARTICLE by a food magazine.\n\n" + doc.text
        })

        question = self._prompt("What QUESTION could that ARTICLE helps answer? Respond only with the QUESTION")

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

        summary = self._prompt(
            "Summarize the ARTICLE in three paragraphs to answer that QUESTION. Include all relevant details. "
            "Describe the facts directly without mentioning the ARTICLE they come from.",
            max_tokens=600)

        return self._build_conversation(question, summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model')  # positional argument
    args = parser.parse_args()
    llm = Llama(model_path=args.model, n_gpu_layers=99, n_ctx=8192, chat_format="chatml", verbose=True)

    evaluator = SummarizingTrainer(llm)
    print(evaluator.process_document(text))
