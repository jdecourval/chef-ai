import dataclasses
import datetime
import json
import sqlite3
from typing import Any, Generator

from db_utils import field_description, DataclassIterableMixin
from model import Document, Recipe


class SQLitePipeline:
    def __init__(self):
        sqlite3.enable_callback_tracebacks(True)
        sqlite3.register_adapter(dict, json.dumps)
        sqlite3.register_adapter(list, json.dumps)
        sqlite3.register_adapter(datetime.timedelta, lambda x: x.seconds)
        sqlite3.register_adapter(Document, lambda x: x.id)  # TODO: Make generic

        # Register the adapter and converter
        sqlite3.register_converter("dict", json.loads)
        sqlite3.register_converter("list", json.loads)
        sqlite3.register_converter("timedelta", lambda x: datetime.timedelta(seconds=int(x)))

        def dict_factory(cursor, row):
            fields = [column[0] for column in cursor.description]
            return {key: value for key, value in zip(fields, row)}

        self.connection = sqlite3.connect('results.db', detect_types=sqlite3.PARSE_DECLTYPES)
        self.connection.execute("PRAGMA synchronous = normal")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.autocommit = False
        self.connection.row_factory = dict_factory

        with self.connection:
            self._create_table_from_dataclass(Document)  # TODO: Move out of here to make generic.
            self._create_table_from_dataclass(Recipe)

    def _create_table_from_dataclass(self, dc: type):
        fields = ','.join([field_description(field) for field in dataclasses.fields(dc)])
        self.connection.execute(f"CREATE TABLE IF NOT EXISTS {dc.__name__}({fields})")

    def process_item(self, item: DataclassIterableMixin):
        primary_key = item.primary_key().name
        insert = "INSERT INTO {} ({}) VALUES ({}) RETURNING {}".format(
            type(item).__name__,
            ','.join(item.fields_name()),
            ','.join(item.placeholders()),
            item.primary_key().name
        )

        with self.connection:
            result = self.connection.execute(insert, item)
            setattr(item, primary_key, result.fetchone()[primary_key])

    def select(self, query: str, *args, **kwargs) -> Generator[dict[str, Any], None, None]:
        with self.connection:
            # Does this risk leaving the cursor open if the iterator is not fully iterated?
            for i in self.connection.execute(query, *args, **kwargs):
                yield i
