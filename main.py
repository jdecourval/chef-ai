import argparse
import logging

import anyio
from anyio import run, Semaphore
from tqdm import tqdm

from ai.engine import ExLlama, LlamaCppPython
from db.db import SQLitePipeline
from indexer.indexer import Indexer
from spider.spider import start as spider_start
from trainer.recipe_evaluator import RecipeEvaluatorTrainer
from trainer.recipe_trainer import RecipeTrainer
from trainer.summarizing_trainer import SummarizingTrainer
from utils.aenumerate import aenumerate

_logger = logging.getLogger(__name__)

quick = False
document_parallelism = 12


async def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    args = parser.parse_args()

    _logger.info("Starting spider")
    await spider_start()

    # _logger.info("Starting enrichment")
    # await spider_enrich()  # Should not be necessary.

    _logger.info("Setting up DB")
    sql = SQLitePipeline()

    _logger.info("Indexing")
    Indexer(sql).start()

    _logger.info("Loading LLM")
    if args.model.endswith(".gguf"):
        llm = LlamaCppPython(model=args.model)
    else:
        llm = ExLlama(model=args.model)

    semaphore = Semaphore(document_parallelism)
    revision = llm.model_name()

    async with llm:
        async with anyio.create_task_group() as tg:
            for trainer_type in RecipeEvaluatorTrainer, SummarizingTrainer, RecipeTrainer:
                count = 0
                _logger.info(f"Starting trainer: {trainer_type.__name__}")
                with tqdm(total=trainer_type.total_document(sql, revision=revision, quick=quick)) as progress:
                    async for count, document in aenumerate(trainer_type.document_generator(sql, quick=quick)):
                        # TODO: Transaction per document
                        await semaphore.acquire()
                        trainer = trainer_type(document, llm, revision=revision)

                        async def process():
                            try:
                                async for training in trainer:
                                    sql.insert(training)
                            except:
                                _logger.error(f"Failed to process document {document}", exc_info=True)
                            semaphore.release()
                            progress.update()

                        tg.start_soon(process)
                        await anyio.sleep(0)  # yield
                _logger.info(f"Done with {trainer_type.__name__}. Created {count} trainings.")

    _logger.info("Finetuning")
    finetuning = Finetuning(sql)
    finetuning.train()

    _logger.info("Dequantizing")

    _logger.info("Merging LoRA")
    _logger.info("Converting weights to Llama.cpp")
    _logger.info("Quantizing")

if __name__ == '__main__':
    run(main)
