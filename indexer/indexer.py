import json
import logging
import os
import sqlite3
from pathlib import Path

from tqdm import tqdm

from db.db import SQLitePipeline
from model.model import Document


_logger = logging.getLogger(__name__)


class Indexer:
    def __init__(self, sql: SQLitePipeline):
        self._sql = sql
        self.indexed = set(self._sql.select_one_col("SELECT source from Document"))

    def insert_db(self, document: Document):
        self._sql.insert(document)
        if document.recipe:
            self._sql.insert(document.recipe)

    def index_html(self, html, source):
        try:
            document = Document.from_html(html, source)
        except json.decoder.JSONDecodeError as e:
            # Sometimes, there's a " not properly escaped that causes parsing to fail.
            if "Expecting ',' delimiter" in e.msg:
                _logger.debug("JSON document has unescaped quotes. Ignoring this one.")
            else:
                _logger.exception("Exception with", e)
        except Exception as e:
            _logger.exception(f"Exception in: {source}, with", e)
        else:
            try:
                self.insert_db(document)
            except sqlite3.IntegrityError:
                # There are actually some non-duplicated documents that have the same names. Too bad for them.
                # .e.g. Gluten-Free Snickerdoodles Recipe, Confetti Cookies Recipe.
                _logger.info(f"Document already indexed: {document}")

    def index_path(self, path):
        if path in self.indexed:
            _logger.info(f"Skipped already parsed document: {path}")
            return
        with open(path, "r") as file:
            html = file.read()
        if not html:
            _logger.error(f"Found a corrupted file, removing it: {path}")
            os.remove(path)
            return
        self.index_html(html, path)

    def start(self, results_folder="results"):
        for path in tqdm(list(Path(results_folder).iterdir())):
            self.index_path(path)


if __name__ == '__main__':
    sql = SQLitePipeline()
    indexer = Indexer(sql)
    indexer.start()
