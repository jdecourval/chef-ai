import argparse
import logging

from anyio import run
from llama_cpp import Llama, LlamaRAMCache

from db.db import SQLitePipeline
from formatter.formatter import ChatMLFormatter
from indexer.indexer import Indexer
from spider.spider import start as spider_start
from spider.spider import enrich as spider_enrich
from trainer.trainer import RecipeEvaluatorTrainer, RecipeTrainer, SummarizingTrainer

_logger = logging.getLogger(__name__)

quick = True


async def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    args = parser.parse_args()

    _logger.info("Starting spider")
    await spider_start()

    _logger.info("Starting enrichment")
    await spider_enrich()  # Should not be necessary.

    _logger.info("Setting up DB")
    sql = SQLitePipeline()

    _logger.info("Starting indexer")
    Indexer(sql).start()

    _logger.info("Loading LLM")
    llm = Llama(model_path=args.model, n_gpu_layers=99, n_ctx=16 * 1024, chat_format="chatml", verbose=False,
                embedding=True)
    llm.set_cache(LlamaRAMCache(100 * 1024 ** 2))

    for trainer in RecipeEvaluatorTrainer, RecipeTrainer, SummarizingTrainer:
        _logger.info(f"Starting trainer: {trainer.__name__}")
        trainer(llm, sql, limit=quick).start()

    _logger.info("Generating training dataset")
    formatter = ChatMLFormatter(sql)
    formatter.start()


if __name__ == '__main__':
    run(main)
