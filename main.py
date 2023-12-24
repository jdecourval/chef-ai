import argparse
import logging

from anyio import run

from ai.engine import ExLlama, LlamaCppServer
from db.db import SQLitePipeline
from finetuning.finetuning import Finetuning
from indexer.indexer import Indexer
from spider.spider import start as spider_start
from trainer.recipe_evaluator import RecipeEvaluatorTrainer
from trainer.recipe_trainer import RecipeTrainer
from trainer.summarizing_trainer import SummarizingTrainer

_logger = logging.getLogger(__name__)

quick = False


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
        llm = LlamaCppServer(model=args.model)
    else:
        llm = ExLlama(model=args.model)

    async with llm:
        for trainer in RecipeEvaluatorTrainer, RecipeTrainer, SummarizingTrainer:
            _logger.info(f"Starting trainer: {trainer.__name__}")
            await trainer(llm, sql, limit=quick).start()

    _logger.info("Finetuning")
    finetuning = Finetuning(sql)
    finetuning.train()

    _logger.info("Dequantizing")

    _logger.info("Merging LoRA")
    _logger.info("Converting weights to Llama.cpp")
    _logger.info("Quantizing")

if __name__ == '__main__':
    run(main)
