import dataclasses
import sqlite3

from utils.generator import first


def field_description(field: dataclasses.Field):
    return '{} {} {} {}'.format(
        field.name,
        field_type(field),
        field_nullable(field),
        field.metadata or ""
    )


def field_type(field: dataclasses.Field):
    if field.type is int:
        return "INTEGER"
    if field.type is float:
        return "REAL"
    if field.type is bool:
        return "INTEGER"
    if field.type is str:
        return "TEXT"
    if isinstance(field.type, type) and issubclass(field.type, DataclassIterableMixin):
        return (f"{field_type(field.type.primary_key_field())} "
                f"REFERENCES {field.type.__name__}({field.type.primary_key_name()})")
    # This works together with register_converter.
    return field.type.__name__  # TODO: Check that the type has been registered.


def field_nullable(field: dataclasses.Field):
    return "" if field.default is None else "NOT NULL"


# noinspection PyDataclass
class DataclassIterableMixin:
    @property
    def _fields_values(self):
        """Cache the most common operations"""
        try:
            return self._non_null_fields_data
        except AttributeError:
            self._non_null_fields_data = [getattr(self, field) for field in self.fields_name()]
            return self._non_null_fields_data

    def __iter__(self):
        yield from self._fields_values

    def __getitem__(self, item):
        return self._fields_values[item]

    def __len__(self):
        return len(self._fields_values)

    def __conform__(self, protocol):
        if protocol is sqlite3.PrepareProtocol:
            return self.primary_key

    def fields_name(self):
        return [field.name for field in dataclasses.fields(self) if getattr(self, field.name) is not None]

    def placeholders(self):
        return ["?"
                for field in dataclasses.fields(self)
                if getattr(self, field.name) is not None]

    @classmethod
    def primary_key_field(cls):
        return first(field for field in dataclasses.fields(cls) if "PRIMARY KEY" in field.metadata)

    @classmethod
    def primary_key_name(cls):
        try:
            return cls.__primary_key_name
        except AttributeError:
            cls.__primary_key_name = cls.primary_key_field().name
            return cls.__primary_key_name

    @property
    def primary_key(self):
        return getattr(self, self.primary_key_name())

    @primary_key.setter
    def primary_key(self, value):
        setattr(self, self.primary_key_name(), value)
