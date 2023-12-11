import contextlib
import enum
import json
from dataclasses import field, dataclass
from datetime import timedelta
from html import unescape

import isodate
import numpy as np
from parsel import Selector

from db_utils import DataclassIterableMixin


@dataclass
class Document(DataclassIterableMixin):
    # TODO: Move somewhere else
    @staticmethod
    def from_html(html: str):
        selector = Selector(html)
        metadata = json.loads(selector.css("script#schema-lifestyle_1-0::text").get(), strict=False)[0]

        self = Document(
            title=unescape(metadata["headline"]),
            text=unescape("".join(selector.css("article .text-passage p.comp::text").getall()).strip()),
            subtitle=unescape(selector.xpath('//meta[@name="description"]/@content').get()),
            author=unescape(metadata["author"][0]["name"]),
        )
        self.recipe = Recipe.build(self, metadata) if metadata["@type"][0] == "Recipe" else None
        assert self.text, "Empty document"
        return self

    title: str
    text: str
    author: str
    subtitle: str = None
    id: int = field(metadata="PRIMARY KEY", default=None)


@dataclass
class Recipe(DataclassIterableMixin):
    # TODO: Move somewhere else
    @staticmethod
    def build(document: Document, metadata: dict):
        self = Recipe(document=document)
        self.document = document

        with contextlib.suppress(KeyError):
            self.review_score = float(metadata["aggregateRating"]["ratingValue"])
        with contextlib.suppress(KeyError):
            self.review_count = int(metadata["aggregateRating"]["ratingCount"])
        with contextlib.suppress(KeyError):
            self.reviews = [unescape(i["reviewBody"]) for i in metadata["review"]]
        with contextlib.suppress(KeyError):
            self.ingredients = unescape(metadata["recipeIngredient"])
        with contextlib.suppress(KeyError):
            self.directions = [unescape(i["text"]) for i in metadata["recipeInstructions"]]
        with contextlib.suppress(KeyError):
            self.nutrition = {i: unescape(j) for i, j in metadata["nutrition"].items() if i != "@type"}
        with contextlib.suppress(KeyError):
            self.category = [unescape(i) for i in metadata["recipeCategory"]]
        with contextlib.suppress(KeyError):
            self.cuisine = [unescape(i) for i in metadata["recipeCuisine"]]
        with contextlib.suppress(KeyError, TypeError):
            self.prep_time = isodate.parse_duration(metadata["prepTime"])
        with contextlib.suppress(KeyError, TypeError):
            self.total_time = isodate.parse_duration(metadata["totalTime"])
        with contextlib.suppress(KeyError):
            self.recipeYield = unescape(metadata["recipeYield"])

        return self

    document: Document
    ingredients: list[str] = field(default_factory=list)
    directions: list[str] = field(default_factory=list)
    nutrition: dict[str, str] = field(default_factory=dict)
    review_score: float = None  # /5
    review_count: int = 0
    reviews: list[str] = field(default_factory=list)
    category: list[str] = field(default_factory=list)
    cuisine: list[str] = field(default_factory=list)
    prep_time: timedelta = None
    total_time: timedelta = None
    recipeYield: str = None
    id: int = field(metadata="PRIMARY KEY", default=None)

    def format_ingredients(self):
        return "\n".join(self.ingredients)

    def format_directions(self):
        return "\n".join(f"{idx + 1}. {i}" for idx, i in enumerate(self.directions))

    def format_nutrition(self):
        return "\n".join(f"{i}: {j}" for i, j in self.nutrition.items())

    def format_reviews(self):
        return "\n".join(self.reviews)

    def __repr__(self):
        return f"""Ingredients:
{self.format_ingredients()}

Instructions:
{self.format_directions()}
"""


@dataclass
class Training(DataclassIterableMixin):
    class Role(enum.IntEnum):
        system = 1
        user = 2
        assistant = 3

    conversation: int  # messages part of the same conversation have the same id.
    position: int  # position in the conversation.
    content: str
    role: Role
    embedding: np.ndarray  # Serialized numpy array.
    trainer: str  # The class that generated this content.
    source: Document = None  # The document that helped generated this content, if applicable. For bookkeeping.
    id: int = field(metadata="PRIMARY KEY", default=None)
