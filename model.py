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
    def __init__(self, html):
        selector = Selector(html)
        self.id = uuid.uuid4()
        self.json = json.loads(selector.css("script#schema-lifestyle_1-0::text").get(), strict=False)[0]
        self.html_fragment = selector.css("article .text-passage").get()
        self.title = unescape(self.json["headline"])
        self.text = unescape("".join(selector.css("article .text-passage p.comp::text").getall()).strip())
        self.subtitle = unescape(selector.xpath('//meta[@name="description"]/@content').get())
        self.author = unescape(self.json["author"][0]["name"])
        self.recipe = Recipe(self) if self.json["@type"][0] == "Recipe" else None
        assert self.text, "Empty document"

    id: uuid.UUID = field(metadata="PRIMARY KEY")
    json: dict = field(repr=False)  # jsonb.
    html_fragment: str = field(
        repr=False)  # article tag. So that an AI can interpret HTML as additional context (e.g. subtitles).
    title: str
    text: str
    author: str
    subtitle: str = None


@dataclass
class Recipe(DataclassIterableMixin):
    def __init__(self, document: Document):
        self.document = document
        self.id = uuid.uuid4()

        try:
            self.review_score = float(document.json["aggregateRating"]["ratingValue"])
        except:
            pass
        try:
            self.review_count = int(document.json["aggregateRating"]["ratingCount"])
        except:
            pass
        try:
            self.ingredients = unescape(document.json["recipeIngredient"])
        except:
            self.ingredients = []
        try:
            self.directions = [unescape(i["text"]) for i in document.json["recipeInstructions"]]
        except:
            self.directions = []
        try:
            self.nutrition = {i: unescape(j) for i, j in document.json["nutrition"].items() if i != "@type"}
        except:
            self.nutrition = {}
        try:
            self.category = unescape(document.json["recipeCategory"])
        except:
            pass
        try:
            self.cuisine = unescape(document.json["recipeCuisine"])
        except:
            pass
        try:
            self.prep_time = isodate.parse_duration(document.json["prepTime"]) if "prepTime" in document.json else None
        except:
            pass
        try:
            self.total_time = isodate.parse_duration(document.json["totalTime"])
        except:
            pass
        try:
            self.recipeYield = unescape(document.json["recipeYield"])
        except:
            pass

    id: uuid.UUID = field(metadata="PRIMARY KEY")
    document: Document
    ingredients: list[str]
    directions: list[str]
    nutrition: dict[str, str]
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
