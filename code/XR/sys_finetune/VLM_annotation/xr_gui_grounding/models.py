from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class ObjectObservation(BaseModel):
    name: str
    description: str
    color: Optional[str] = None
    position: str


class SceneUnderstanding(BaseModel):
    scene: str
    activity: str
    objects: List[ObjectObservation]


class WindowPosition(BaseModel):
    description: str
    bbox_norm: List[float] = Field(min_length=4, max_length=4)

    @field_validator("bbox_norm")
    @classmethod
    def validate_bbox_norm(cls, value: List[float]) -> List[float]:
        if len(value) != 4:
            raise ValueError("bbox_norm must have 4 values")
        x1, y1, x2, y2 = value
        if not (0 <= x1 < x2 <= 1) or not (0 <= y1 < y2 <= 1):
            raise ValueError("bbox_norm values must be within [0,1] and x1<x2, y1<y2")
        return value


class UiCandidate(BaseModel):
    app_name: str
    reason: str
    required_items: List[str]
    ui_elements: List[str]
    window_position: WindowPosition


class UiCandidates(BaseModel):
    apps: List[UiCandidate]


class ImagePrompt(BaseModel):
    prompt: str


class Annotation(BaseModel):
    ui_element: str
    direct_instruction: str
    semantic_instruction: str
    direct_instruction_en: Optional[str] = None
    semantic_instruction_en: Optional[str] = None
    related_object: Optional[str] = None


class Annotations(BaseModel):
    annotations: List[Annotation]


@dataclass
class PipelineConfig:
    output_dir: str
    vlm_model: str
    max_apps: int
    app_index: int
    image_model: str
    image_size: str
    skip_image_gen: bool
    skip_annotations: bool
    xr_image: Optional[str]
    instruction_language: str
    enable_thinking: bool
    visualize_annotations: bool
    num_annotations: int
    bilingual_annotations: bool
    max_images: int
