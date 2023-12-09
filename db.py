import dataclasses
import datetime
import json
import sqlite3
import uuid

from db_utils import field_description, DataclassIterableMixin
from model import Document, Recipe


class SQLitePipeline:
    def __init__(self):
        sqlite3.enable_callback_tracebacks(True)
        sqlite3.register_adapter(dict, json.dumps)
        sqlite3.register_adapter(list, json.dumps)
        sqlite3.register_adapter(uuid.UUID, lambda x: x.bytes)
        sqlite3.register_adapter(datetime.timedelta, lambda x: x.seconds)
        sqlite3.register_adapter(Document, lambda x: x.id.bytes)  # TODO: Make generic

        # Register the adapter and converter
        sqlite3.register_converter("dict", json.loads)
        sqlite3.register_converter("list", json.loads)

        self.connection = sqlite3.connect('results.db', autocommit=False)  # TODO: autocommit=False on Python 3.12

        with self.connection:
            self.connection.execute("PRAGMA journal_mode = WAL")
            self.connection.execute("PRAGMA foreign_keys = ON")
            self.connection.execute("PRAGMA synchronous = normal")
            self._create_table_from_dataclass(Document)  # TODO: Move out of here to make generic.
            self._create_table_from_dataclass(Recipe)

    def _create_table_from_dataclass(self, dc: type):
        fields = ','.join([field_description(field) for field in dataclasses.fields(dc)])
        self.connection.execute(f"CREATE TABLE IF NOT EXISTS {dc.__name__}({fields})")

    def process_item(self, item: DataclassIterableMixin):
        insert = "INSERT INTO {} ({}) VALUES ({})".format(
            type(item).__name__,
            ','.join(item.fields_name()),
            ','.join(item.placeholders()),
        )

        with self.connection:
            self.connection.execute(insert, item)
