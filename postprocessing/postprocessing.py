from db.db import SQLitePipeline


class Postprocessing:
    pass

class RemoveLookalikes(Postprocessing):
    def __init__(self, sql: SQLitePipeline):
        self._sql = sql
