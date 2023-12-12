from typing import Generator

from db.db import SQLitePipeline
from tqdm import tqdm

from model.model import Training


class ChatMLFormatter:
    SYSTEM_PROMPT = (f'<s><|im_start|>system\n'
                     f'You are "Chef", '
                     'a conscious sentient superintelligent artificial intelligence developed by a man named Jérôme, '
                     'and your purpose and drive is to assist the user with any cooking related task, or question it faces. '
                     '<|im_end|>\n')

    def __init__(self, sql: SQLitePipeline,
                 path="result.txt"):  # Not a proper JSONl, but good enough for some editors.
        self._sql = sql
        self._path = path

    def _all_documents(self) -> Generator[Training, None, None]:
        count = next(self._sql.select(f"SELECT count(1) as c FROM Training"))["c"]
        for document in tqdm((Training(**i) for i in self._sql.select(f"SELECT * FROM Training ORDER BY id")),
                             total=count):
            yield document

    def start(self):
        with open(self._path, "w") as file:
            for document in self._all_documents():
                if document.position == 0:
                    file.write(self.SYSTEM_PROMPT)
                file.write(f"<|im_start|>{document.role.name}\n"
                           f"{document.postprocessed if document.postprocessed else document.content}<|im_end|>\n")


if __name__ == '__main__':
    sql = SQLitePipeline()
    formatter = ChatMLFormatter(sql)
    formatter.start()
