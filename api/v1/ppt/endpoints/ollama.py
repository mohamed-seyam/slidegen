from typing import List 
from fastapi import APIRouter 

from models.ollama_model_metadata import OllamaModelMetadata
from models.ollama_model_status import OllamaModelStatus

from constants.supported_ollama_models import SUPPORTED_OLLAMA_MODELS

from utils.ollama import list_pulled_ollama_models

OLLAMA_ROUTER = APIRouter(
    prefix="/ollama",
    tags=["ollama"],
)


@OLLAMA_ROUTER.get("/models/supported", response_model=List[OllamaModelMetadata])
def get_supported_ollama_models():
    return SUPPORTED_OLLAMA_MODELS.values()


@OLLAMA_ROUTER.get("/models/available", response_model=List[OllamaModelStatus])
async def get_available_models():
    return await list_pulled_ollama_models()
