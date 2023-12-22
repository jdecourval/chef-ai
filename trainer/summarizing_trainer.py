import itertools
import logging
import re
from typing import override, Generator, AsyncGenerator

import numpy as np
from anyio import create_task_group
from llama_cpp import LlamaGrammar
from sentence_transformers.util import cos_sim
from tqdm import tqdm

from model.model import Training, Document
from trainer.trainer import Trainer, main
from utils.aenumerate import aenumerate

_logger = logging.getLogger(__name__)


class SummarizingTrainer(Trainer):
    GRAMMAR_KNOWLEDGE = LlamaGrammar.from_string('root ::= "anecdotes" | "story" | "knowledge"', verbose=False)
    # MIN_DOC_SIZE_B = 500  # Largest is 53453, avg is 5667.
    MIN_DOC_SIZE_B = 8000  # This gives 1379 documents which is way more manageable for now.

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log = open("summarizing_trainer.log", "w")

    def _documents_without_recipe(self) -> Generator[Document, None, None]:
        # octet_length is faster for an approximate length.
        total = next(self._sql.select_one_col("SELECT count(1) as c FROM Document "
                                              "LEFT OUTER JOIN Recipe ON (Document.id = Recipe.document) "
                                              "WHERE Recipe.document IS NULL "
                                              f"AND octet_length(Document.text) > {self.MIN_DOC_SIZE_B}"))
        for document in tqdm((Document(**i) for i in self._sql.select(
                "SELECT Document.* FROM Document "
                "LEFT JOIN Recipe ON (Document.id = Recipe.document) "
                f"WHERE Recipe.document IS NULL "
                f"AND octet_length(Document.text) > {self.MIN_DOC_SIZE_B} {self._limit}")),
                             total=total):
            yield document

    @override
    async def __aiter__(self) -> AsyncGenerator[Training, None]:
        last_index = next(self._sql.select_one_col(
            f"SELECT coalesce(MAX(conversation), 0) FROM Training WHERE trainer='{self.__class__.__name__}'"))
        for idx, document in enumerate(self._documents_without_recipe(), start=last_index + 1):
            if next(self._sql.select_one_col(
                    "SELECT count(1) FROM Training WHERE source=? AND trainer='SummarizingTrainer'", (document,))):
                _logger.info(f"Skipping over already processed document: {document}")
                continue

            try:
                async for position, conversation in aenumerate(self._process_document(document)):
                    yield self._training(conversation=conversation,
                                         conversation_id=idx,
                                         position=position % 2,  # Assumes _process_document generates many q&a.
                                         source=document
                                         )
            except Exception as e:
                _logger.exception(f"Failed to process recipe: {document.title}", e)

    async def _process_document(self, doc: Document) -> AsyncGenerator[dict[str, str], None]:
        with self.chat_scope():
            self.chat.append({
                "role": "user",  # Using system breaks the next prompt.
                "content": "Starting after the line break is an ARTICLE by a food magazine.\n\n" + doc.text
            })

            with self.chat_scope():
                if await self.chat.chat(
                        "Does the ARTICLE talks of anecdotes, does it tell a story, or is it about culinary knowledge? "
                        "Your answer must be one word: 'anecdotes', 'story' or 'knowledge'.",
                        grammar=self.GRAMMAR_KNOWLEDGE
                ) in ["anecdotes", "story"]:
                    return

            # TODO: Use grammar
            with self.chat_scope():
                questions = [re.match(r"[\d.-]* (.*)", i)[1] for i in itertools.filterfalse(
                    lambda line: re.search(r"in the recipe|article|this|that|these|those|author|her|his", line),
                    (await self.chat.chat(
                        "Are there general cooking-knowledge related QUESTION(s) that the content of this ARTICLE would answer? "
                        "Your response must not refer to anything from the ARTICLE. "
                        'Format your response as a list of zero to twenty QUESTION(s) which must end with a question mark(?). '
                        'These words are banned from your response: author, this, that, these, those, article. '
                        "Now forget about the ARTICLE. "
                        "Each QUESTION in your response must stand independently of the ARTICLE. "
                        "Remove any QUESTION from your response where it's not the case. ")).splitlines())]

                # TODO: Pick more questions in larger documents
                while len(questions) > 6:
                    embeddings = self.embed_model.encode(questions, show_progress_bar=False, normalize_embeddings=True)
                    mean = np.mean(embeddings, axis=0)
                    entropy = sorted(((i[0], cos_sim(i[1], mean)) for i in enumerate(embeddings)), key=lambda x: x[1])
                    _logger.info(f"Dropped question: {questions[entropy[-1][0]]}")
                    del questions[entropy[-1][0]]

            results = []
            async def gen_q_and_a(question):
                with self.chat_scope():
                    summary = await self.chat.chat(
                        f'Respond, in your own words, not the author\'s, to the question "{question}". '
                        'Act like you never saw the article. '
                        f'Explain step-by-step, miticulously. Respond only with your long and detailed response. '
                        f'Elaborate in up to 600 words. '
                        f'These words are banned from your response: author, article. ',
                        max_tokens=2000)

                    results.extend(self._q_and_q_messages(question, summary))

            async with create_task_group() as tg:
                for question in questions:
                    tg.start_soon(gen_q_and_a, question)

            for result in results:
                yield result


if __name__ == '__main__':
    main(SummarizingTrainer, False)
