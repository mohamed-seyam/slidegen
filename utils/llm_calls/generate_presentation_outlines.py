from typing import Optional

from utils.llm_provider import get_model
from utils.get_dynamic_models import get_presentation_outline_model_with_n_slides
from models.presentation_outline_model import PresentationOutlineModel

from services.llm_client import LLM_Client

async def generate_ppt_outline(
        content: str, 
        n_slides: int, 
        language: Optional[str] = None,
        additional_context: Optional[str] = None,
        tone: Optional[str]  = None,
        verbosity: Optional[str] = None,
        instructions: Optional[str] = None,
        include_title_slide: bool = True,
        web_search: bool = False
):
    model = get_model()
    reponse_model = get_presentation_outline_model_with_n_slides(n_slides=n_slides)

    client = LLM_Client()

    try:
        async for chunk in client.generate_structured()
    