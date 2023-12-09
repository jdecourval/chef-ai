import dataclasses
import datetime


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
    if issubclass(field.type, DataclassIterableMixin):
        return f"{field_type(field.type.primary_key())} REFERENCES {field.type.__name__}({field.type.primary_key().name})"
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

    def fields_name(self):
        return [field.name for field in dataclasses.fields(self) if getattr(self, field.name) is not None]

    def placeholders(self):
        return ["?"
                for field in dataclasses.fields(self)
                if getattr(self, field.name) is not None]

    @classmethod
    def primary_key(cls):
        return next(field for field in dataclasses.fields(cls) if "PRIMARY KEY" in field.metadata)
