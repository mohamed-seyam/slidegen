from fastapi import APIRouter, Body, HTTPException
from typing import Annotated

from utils.available_models import list_available_qwen_compatible_models
QWEN_ROUTER = APIRouter(
    prefix="/qwen",
    tags=["qwen"],
)

@QWEN_ROUTER.post("/models/available")
async def get_available_models(
    url: Annotated[str, Body()],
    api_key: Annotated[str, Body()]
    ):
    try:
        return await list_available_qwen_compatible_models(url, api_key)
    
    except Exception as e:
        return HTTPException(status_code=500, detail=str(e))
    


