import dataclasses
import datetime
import json
import sqlite3
from typing import Any, Generator, Type

from db.db_utils import field_description, DataclassIterableMixin
from model.model import Document, Recipe, Training


class SQLitePipeline:
    def __init__(self):
        sqlite3.enable_callback_tracebacks(True)
        sqlite3.register_adapter(dict, json.dumps)
        sqlite3.register_adapter(list, json.dumps)
        sqlite3.register_adapter(datetime.timedelta, lambda x: x.seconds)
        sqlite3.register_adapter(Document, lambda x: x.primary_key)

        # Register the adapter and converter
        sqlite3.register_converter("dict", json.loads)
        sqlite3.register_converter("list", json.loads)
        sqlite3.register_converter("timedelta", lambda x: datetime.timedelta(seconds=int(x)))
        sqlite3.register_converter("Role", lambda x: Training.Role(int(x)))  # TODO: Make generic

        def dict_row_factory(cursor, row):
            fields = [column[0] for column in cursor.description]
            return {key: value for key, value in zip(fields, row)}

        self.connection = sqlite3.connect('results.db', detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
        self.connection.execute("PRAGMA synchronous = normal")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.autocommit = False
        self.connection.row_factory = dict_row_factory

        with self.connection:
            self._create_table_from_dataclass(Document)  # TODO: Move out of here to make generic.
            self._create_table_from_dataclass(Recipe)
            self._create_table_from_dataclass(Training)
            # This will help pulling training solutions from the DB in the corect order.
            self.connection.execute(
                "CREATE INDEX IF NOT EXISTS training_index ON training (trainer, conversation, position)")

    def _create_table_from_dataclass(self, dc: Type[dataclasses.dataclass]):
        fields = ','.join([field_description(field) for field in dataclasses.fields(dc)])
        self.connection.execute(f"CREATE TABLE IF NOT EXISTS {dc.__name__}({fields})")

    def insert(self, item: DataclassIterableMixin):
        insert = "INSERT INTO {} ({}) VALUES ({}) RETURNING {}".format(
            type(item).__name__,
            ','.join(item.fields_name()),
            ','.join(item.placeholders()),
            item.primary_key_name()
        )

        with self.connection:
            result = self.connection.execute(insert, item)
            item.primary_key = result.fetchone()[item.primary_key_name()]

    def select(self, query: str, *args, **kwargs) -> Generator[dict[str, Any], None, None]:
        with self.connection:
            # Does this risk leaving the cursor open if the iterator is not fully iterated?
            for i in self.connection.execute(query, *args, **kwargs):
                yield i

    def select_one_col(self, query: str, *args, **kwargs) -> Generator[Any, None, None]:
        backup = self.connection.row_factory
        try:
            with self.connection:
                # TODO: Any way to avoid doing this?
                self.connection.row_factory = lambda cursor, row: row[0]
                # Does this risk leaving the cursor open if the iterator is not fully iterated?
                for i in self.connection.execute(query, *args, **kwargs):
                    yield i
        finally:
            self.connection.row_factory = backup
