import contextlib
import json
import uuid
from dataclasses import field, dataclass
from datetime import timedelta
from html import unescape

import isodate
from parsel import Selector

from db_utils import DataclassIterableMixin


@dataclass
class Document(DataclassIterableMixin):
    @staticmethod
    def from_html(html):
        selector = Selector(html)
        metadata = json.loads(selector.css("script#schema-lifestyle_1-0::text").get(), strict=False)[0]

        self = Document(
            json=metadata,
            html_fragment=selector.css("article .text-passage").get(),
            title=unescape(metadata["headline"]),
            text=unescape("".join(selector.css("article .text-passage p.comp::text").getall()).strip()),
            subtitle=unescape(selector.xpath('//meta[@name="description"]/@content').get()),
            author=unescape(metadata["author"][0]["name"]),
        )
        self.recipe = Recipe.from_document(self) if metadata["@type"][0] == "Recipe" else None
        assert self.text, "Empty document"
        return self

    json: dict = field(repr=False)  # jsonb.
    html_fragment: str = field(
        repr=False)  # article tag. So that an AI can interpret HTML as additional context (e.g. subtitles).
    title: str
    text: str
    author: str
    subtitle: str = None
    id: uuid.UUID = field(metadata="PRIMARY KEY", default_factory=uuid.uuid4)


@dataclass
class Recipe(DataclassIterableMixin):
    @staticmethod
    def from_document(document: Document):
        self = Recipe(document=document)
        self.document = document
        self.id = uuid.uuid4()

        with contextlib.suppress(KeyError):
            self.review_score = float(document.json["aggregateRating"]["ratingValue"])
        with contextlib.suppress(KeyError):
            self.review_count = int(document.json["aggregateRating"]["ratingCount"])
        with contextlib.suppress(KeyError):
            self.ingredients = unescape(document.json["recipeIngredient"])
        with contextlib.suppress(KeyError):
            self.directions = [unescape(i["text"]) for i in document.json["recipeInstructions"]]
        with contextlib.suppress(KeyError):
            self.nutrition = {i: unescape(j) for i, j in document.json["nutrition"].items() if i != "@type"}
        with contextlib.suppress(KeyError):
            self.category = unescape(document.json["recipeCategory"])
        with contextlib.suppress(KeyError):
            self.cuisine = unescape(document.json["recipeCuisine"])
        with contextlib.suppress(KeyError, TypeError):
            self.prep_time = isodate.parse_duration(document.json["prepTime"])
        with contextlib.suppress(KeyError, TypeError):
            self.total_time = isodate.parse_duration(document.json["totalTime"])
        with contextlib.suppress(KeyError):
            self.recipeYield = unescape(document.json["recipeYield"])

        return self

    id: uuid.UUID = field(metadata="PRIMARY KEY", default_factory=uuid.uuid4, init=False)
    document: Document
    ingredients: list[str] = field(default_factory=list)
    directions: list[str] = field(default_factory=list)
    nutrition: dict[str, str] = field(default_factory=dict)
    review_score: float = None  # /5
    review_count: int = None
    category: str = None
    cuisine: str = None
    prep_time: timedelta = None
    total_time: timedelta = None
    recipeYield: str = None

    def format_directions(self):
        return "\n".join(f"{idx + 1}. {i}" for idx, i in enumerate(self.directions))

    def format_nutrition(self):
        raise NotImplementedError()

    def __repr__(self):
        return f"""Ingredients:
{self.ingredients}
Instructions:
{self.format_directions()}
"""
