import ast
import re
from abc import ABC
from hashlib import sha256
from io import BytesIO
from typing import Literal

from PIL import Image
from pydantic import BaseModel, field_validator, model_validator

from vdpo.types.enums import DatasetSplitName
from vdpo.utils.inflect_engine import get_inflect_engine

object_name_pattern = re.compile(r"^(.+?)(?: \((.+)\))?$")


class DatasetImage(BaseModel):
    bytes: bytes

    @property
    def pil_image(self) -> Image.Image:
        return Image.open(BytesIO(self.bytes)).convert("RGBA")

    model_config = {
        "ser_json_bytes": "base64",
    }


class DatasetExample(BaseModel, ABC):
    dataset_split_name: DatasetSplitName

    object: str

    original_image: DatasetImage
    counterfact_image: DatasetImage

    correct_answer: list[str] | str
    incorrect_answer: str

    @property
    def example_id(self) -> str:
        h1 = self.object.encode("utf-8")
        h2 = sha256(self.original_image.bytes).digest()
        h3 = sha256(self.counterfact_image.bytes).digest()
        return sha256(h1 + h2 + h3).hexdigest()

    @model_validator(mode="after")
    def validate_object(self):
        self.object = _validate_object_name(self.object)
        return self


class DatasetExampleColor(DatasetExample):
    dataset_split_name: Literal[DatasetSplitName.COLOR] = DatasetSplitName.COLOR

    correct_answer: list[str]

    @field_validator("correct_answer", mode="before")
    @classmethod
    def parse_correct_answer(cls, v):
        if isinstance(v, str):
            try:
                return ast.literal_eval(v)
            except (SyntaxError, ValueError):
                pass
        return v

    @model_validator(mode="after")
    def fix_grey_gray(self):
        self.correct_answer = list(map(_sanitize_color_name, self.correct_answer))
        self.incorrect_answer = _sanitize_color_name(self.incorrect_answer)
        return self

    @model_validator(mode="after")
    def validate_no_color_overlap(self):
        if self.incorrect_answer in self.correct_answer:
            raise ValueError(
                f"Counterfactual color '{self.incorrect_answer}' is also a "
                f"correct color for '{self.object}' — no genuine conflict"
            )
        return self


class DatasetExampleSize(DatasetExample):
    dataset_split_name: Literal[DatasetSplitName.SIZE] = DatasetSplitName.SIZE

    correct_answer: str

    @model_validator(mode="after")
    def validate_answers(self):
        self.correct_answer = _validate_object_name(self.correct_answer)
        self.incorrect_answer = _validate_object_name(self.incorrect_answer)
        return self


def _validate_object_name(name: str) -> str:
    name = _sanitize_object_name(name)

    if name != _strip_object_extra(name):
        raise ValueError(f"Object name '{name}' contains extra information")

    name_singular = get_inflect_engine().singular_noun(name)  # type: ignore

    if name_singular and name_singular != name:
        raise ValueError(f"Object name '{name}' is plural")

    return name


def _strip_object_extra(object_name: str) -> str:
    match = object_name_pattern.match(object_name)

    if not match:
        raise Exception(f"Could not parse object name: {object_name}")

    word, _ = match.groups()
    return word.strip()


def _sanitize_object_name(object_name: str) -> str:
    object_name = object_name.lower().replace("_", " ")
    object_name = " ".join(object_name.split())

    if not object_name:
        raise ValueError("Empty object name")

    match = object_name_pattern.match(object_name)

    if not match:
        raise ValueError(f"Could not parse object name: {object_name}")

    object_name, extra = match.groups()
    object_name = object_name.capitalize()

    if not extra:
        return object_name

    return f"{object_name} ({extra})"


def _sanitize_color_name(color_name: str) -> str:
    color_name = " ".join(color_name.lower().split())

    if not color_name:
        raise ValueError("Empty color name")

    if color_name in {"grey", "gray"}:
        color_name = "Gray"

    return color_name.capitalize()
