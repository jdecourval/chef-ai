import dataclasses
import datetime
import json
import sqlite3
import uuid
from contextlib import contextmanager, nullcontext
from pathlib import PosixPath
from typing import Any, Generator, Type

from db.db_utils import field_description, DataclassIterableMixin
from model.model import Document, Recipe, Training


class SQLitePipeline:
    def __init__(self):
        sqlite3.enable_callback_tracebacks(True)
        sqlite3.register_adapter(dict, json.dumps)
        sqlite3.register_adapter(list, json.dumps)
        sqlite3.register_adapter(datetime.timedelta, lambda x: x.seconds)
        sqlite3.register_adapter(PosixPath, str)
        sqlite3.register_adapter(uuid.UUID, lambda x: x.bytes_le)

        # Register the adapter and converter
        sqlite3.register_converter("dict", json.loads)
        sqlite3.register_converter("list", json.loads)
        sqlite3.register_converter("timedelta", lambda x: datetime.timedelta(seconds=int(x)))
        sqlite3.register_converter("Role", lambda x: Training.Role(int(x)))  # TODO: Make generic
        sqlite3.register_converter("UUID", lambda x: uuid.UUID(bytes_le=x))

        with self._connection() as connection:
            self._create_table_from_dataclass(connection, Document)  # TODO: Move out of here to make generic.
            self._create_table_from_dataclass(connection, Recipe)
            self._create_table_from_dataclass(connection, Training)
            # This will help pulling training solutions from the DB in the correct order.
            connection.execute(
                "CREATE INDEX IF NOT EXISTS training_index ON Training (trainer, conversation, position)")

    @staticmethod
    def _connection():
        def dict_row_factory(cursor, row):
            fields = [column[0] for column in cursor.description]
            return {key: value for key, value in zip(fields, row)}

        connection = sqlite3.connect('results.db', detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
        connection.execute("PRAGMA synchronous = normal")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA foreign_keys = ON")
        connection.autocommit = False
        connection.row_factory = dict_row_factory
        return connection

    @staticmethod
    def _create_table_from_dataclass(connection, dc: Type[dataclasses.dataclass]):
        fields = ','.join([field_description(field) for field in dataclasses.fields(dc)])
        connection.execute(f"CREATE TABLE IF NOT EXISTS {dc.__name__}({fields})")

    def insert(self, item: DataclassIterableMixin, connection=None):
        insert = "INSERT INTO {} ({}) VALUES ({}) RETURNING {}".format(
            type(item).__name__,
            ','.join(item.fields_name()),
            ','.join(item.placeholders()),
            item.primary_key_name()
        )

        with (self._connection() if connection is None else nullcontext(connection)) as connection:
            result = connection.execute(insert, item)
            item.primary_key = result.fetchone()[item.primary_key_name()]

    def select(self, query: str, *args, connection=None, **kwargs) -> Generator[dict[str, Any], None, None]:
        with (self._connection() if connection is None else nullcontext(connection)) as connection:
            for i in connection.execute(query, *args, **kwargs):
                yield i

    def select_one_col(self, query: str, *args, connection=None, **kwargs) -> Generator[Any, None, None]:
        with (self._connection() if connection is None else nullcontext(connection)) as connection:
            connection.row_factory = lambda cursor, row: row[0]
            for i in connection.execute(query, *args, **kwargs):
                yield i

    @contextmanager
    def transaction(self):
        with self._connection() as connection:
            yield connection
