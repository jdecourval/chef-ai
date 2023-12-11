import json
from pathlib import Path

from tqdm import tqdm

from db import SQLitePipeline
from model import Document

# Questions to help generate training dialogs:
"What question could the content of this file helps answer?"

# Questions part of the training dialogs:
# Try multiple versions of each.
"I have walnuts on hand, what recipe could I do?"
"How to cook walnut pie?"
"how much walnut do I need make a pie?"


sqlite = SQLitePipeline()


def insert_db(document: Document):
    sqlite.insert(document)
    if document.recipe:
        sqlite.insert(document.recipe)


def index_html(html):
    try:
        document = Document.from_html(html)
    except json.decoder.JSONDecodeError as e:
        # Sometimes, there's a " not properly escaped that causes parsing to fail.
        if "Expecting ',' delimiter" not in e.msg:
            print("Exception with", e)
    except Exception as e:
        print("Exception with", e)
    else:
        insert_db(document)


def index_path(path):
    with open(path, "r") as file:
        html = file.read()
    if not html:
        print(path, "is corrupted")
        return
    index_html(html)


def start():
    for path in tqdm(list(Path("results").iterdir())):
        index_path(path)


if __name__ == '__main__':
    start()
