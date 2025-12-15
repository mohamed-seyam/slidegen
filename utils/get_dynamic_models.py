
from pydantic import Field
from typing import List 

from models.presentation_outline_model import (
    PresentationOutlineModel, 
    SlideOutlineModel
)

def get_presentation_outline_model_with_n_slides(n_slides: int):
    class SlideOutlineModelWithNSlides(SlideOutlineModel):
        content: str = Field(
            description="Markdown content for each slide",
            min_length=100,
            max_length=100
        )

    class PresentationOutlineModelWithNSlides(PresentationOutlineModel):
        slides: List[SlideOutlineModelWithNSlides] = Field(
            description="List of slide outlines",
            min_items= n_slides,
            max_items = n_slides
        )

    return PresentationOutlineModelWithNSlides


