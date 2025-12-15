from fastapi import APIRouter


from api.v1.ppt.endpoints.qwen import QWEN_ROUTER
from api.v1.ppt.endpoints.files import FILES_ROUTER
from api.v1.ppt.endpoints.outlines import OUTLINES_ROUTER
from api.v1.ppt.endpoints.presentation import PRESENTATION_ROUTER

API_V1_PPT_ROUTER = APIRouter()
API_V1_PPT_ROUTER.include_router(QWEN_ROUTER)
API_V1_PPT_ROUTER.include_router(FILES_ROUTER)
API_V1_PPT_ROUTER.include_router(OUTLINES_ROUTER)
API_V1_PPT_ROUTER.include_router(PRESENTATION_ROUTER)



