import itertools
import logging
import re
from typing import override, AsyncGenerator

import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

from db.db import SQLitePipeline
from model.model import Training, Document
from trainer.trainer import Trainer, main

_logger = logging.getLogger(__name__)


class SummarizingTrainer(Trainer):
    # MIN_DOC_SIZE_B = 500  # Largest is 53453, avg is 5667.
    MIN_DOC_SIZE_B = 8000  # This gives 1379 documents which is way more manageable for now.
    _LIMIT_QUICK = "ORDER BY RANDOM() LIMIT 50"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log = open("summarizing_trainer.log", "w")
        self.grammar_knowledge = self._llm.get_options_grammar(("anecdotes", "story", "knowledge"))
        self.embed_model = SentenceTransformer('thenlper/gte-large')

    @classmethod
    @override
    def total_document(cls, sql: SQLitePipeline, revision, quick=False) -> int:
        if quick:
            return 50
        # octet_length is faster for an approximate length.
        return next(sql.select_one_col(
            "SELECT count(1) as c FROM Document "
            "LEFT OUTER JOIN Recipe ON Document.id=Recipe.document "
            "LEFT JOIN Training ON "
            f"Document.id=Training.source AND trainer='SummarizingTrainer' AND revision='{revision}' "
            "WHERE Recipe.document IS NULL "
            f"AND octet_length(Document.text) > {cls.MIN_DOC_SIZE_B}"))

    @classmethod
    @override
    async def document_generator(cls, sql: SQLitePipeline, revision: str = None,
                                 quick=False) -> AsyncGenerator[Document, None]:
        # For this trainer, only gets article not associated to a recipe. Also skip the ones already done.
        for document in (Document(**i) for i in sql.select(
                "SELECT Document.* FROM Document "
                "LEFT JOIN Recipe ON Document.id=Recipe.document "
                "LEFT JOIN Training ON Document.id=Training.source AND trainer=? AND revision=? "
                f"WHERE Recipe.document IS NULL AND Training.source IS NULL "
                f"AND octet_length(Document.text) > {cls.MIN_DOC_SIZE_B} {cls.LIMIT_QUICK if quick else ''}", (cls.__name__, revision))):
            yield document

    @override
    async def __aiter__(self) -> AsyncGenerator[Training, None]:
        self.chat.append({
            "role": "user",  # Using system breaks the next prompt.
            "content": "Starting after the line break is an ARTICLE by a food magazine.\n\n" + self.input.text
        })

        with self._chat_scope():
            if await self.chat.chat(
                    "Does the ARTICLE talks of anecdotes, does it tell a story, or is it about culinary knowledge? "
                    "Your answer must be one word: 'anecdotes', 'story' or 'knowledge'.",
                    grammar=self.grammar_knowledge
            ) in ["anecdotes", "story"]:
                return

        # TODO: Use grammar
        with self._chat_scope():
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
                _logger.info(f"Dropped question: {questions[entropy[-1][0]]}, over: {questions}")
                del questions[entropy[-1][0]]

        for question in questions:
            with self._chat_scope():
                summary = await self.chat.chat(
                    f'Respond, in your own words, not the author\'s, to the question "{question}". '
                    'Act like you never saw the article. '
                    f'Explain step-by-step, miticulously. Respond only with your long and detailed response. '
                    f'Elaborate in up to 600 words. '
                    f'These words are banned from your response: author, article. ',
                    max_tokens=2000)

                for training in self._q_and_q_training(question, summary):
                    self._new_conversation()
                    yield training


if __name__ == '__main__':
    main(SummarizingTrainer)
