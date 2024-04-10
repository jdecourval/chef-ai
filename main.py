import argparse
import logging
import uuid

import anyio
from anyio import run, Semaphore
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from ai.engine import ExLlama, LlamaCppServer
from db.db import SQLitePipeline
from finetuning.finetuning import Finetuning
from finetuning.merge_qlora import merge_lora
from indexer.indexer import Indexer
from model.model import Training, Recipe
from spider.spider import start as spider_start
from trainer.recipe_evaluator import RecipeEvaluatorTrainer
from trainer.recipe_trainer import RecipeTrainer
from trainer.summarizing_trainer import SummarizingTrainer
from utils.generator import aenumerate

_logger = logging.getLogger(__name__)

quick = False
base_model = 'teknium/OpenHermes-2.5-Mistral-7B'
document_parallelism = 4


async def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    args = parser.parse_args()

    _logger.info("Starting spider")
    await spider_start()

    _logger.info("Setting up DB")
    sql = SQLitePipeline()

    _logger.info("Indexing")
    Indexer(sql).start()

    _logger.info("Loading LLM")
    if args.model.endswith(".gguf"):
        llm = LlamaCppServer(model=args.model)
    else:
        llm = ExLlama(model=args.model)

    # GPU memory may be maxed out already by the LLM.
    embed_model = SentenceTransformer('thenlper/gte-large', device='cpu')
    semaphore = Semaphore(document_parallelism)
    revision = llm.model_name()
    _logger.info(f"Revision: {revision}")

    async with llm, anyio.create_task_group() as tg:
        for trainer_type in RecipeEvaluatorTrainer, SummarizingTrainer, RecipeTrainer:
            count = 0
            _logger.info(f"Starting trainer: {trainer_type.__name__}")
            with tqdm(total=trainer_type.total_document(sql, revision=revision, quick=quick)) as progress:
                async for count, document in aenumerate(
                        trainer_type.document_generator(sql, quick=quick, revision=revision)):
                    await semaphore.acquire()
                    trainer = trainer_type(document, llm, revision=revision, embed_model=embed_model)

                    async def process():
                        try:
                            # Accumulating trainings could be avoided if sqlite supported concurrent writer transactions
                            trainings = [training async for training in trainer]
                            if len(trainings):
                                with sql.transaction() as transaction:
                                    for training in trainings:
                                        sql.insert(training, transaction)
                            else:
                                # Some documents will not yield trainings.
                                # Generate empty trainings for those, so that they are skipped when resuming.
                                empty_training = Training(conversation=uuid.uuid4(), position=0, content="", role=Training.Role.none,
                                         trainer=trainer_type.__name__,
                                         source=document.document if isinstance(document, Recipe) else document,
                                         revision=revision)
                                sql.insert(empty_training)

                                pass
                        except:
                            _logger.error(f"Failed to process document {document}", exc_info=True)
                        semaphore.release()
                        progress.update()

                    tg.start_soon(process)
                    await anyio.sleep(0)  # yield
            _logger.info(f"Done with {trainer_type.__name__}. Created {count} trainings.")

    _logger.info("Finetuning")
    finetuning = Finetuning(sql, "out")
    finetuning.train()
    finetuning.save("out/final")

    _logger.info("Merging LoRA")
    merge_lora(base_model, "out/final", "out/merged", Finetuning.tokenizer(),
               Finetuning.model_init_kwargs)  # TODO: final is not necessarily to best epoch


if __name__ == '__main__':
    run(main)
