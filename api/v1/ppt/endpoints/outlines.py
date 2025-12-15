import uuid
import math 

from fastapi import APIRouter, HTTPException, Depends, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from models.sql.presentation import PresentationModel
from models.sse_response import SSEStatusResponse
from services.database import get_async_session
from services.temp_file import TEMP_FILE_SERVICE
from services.documents_loader import DocumentsLoader

OUTLINES_ROUTER = APIRouter(
    prefix="/outlines",
    tags=["outlines"]
)

@OUTLINES_ROUTER.get("/stream/{id}")
async def stream_outline(
    id: uuid.UUID,
    sql_session: AsyncSession = Depends(get_async_session),
):
    
    presentation = await sql_session.get(PresentationModel, id)

    if not presentation:
        raise HTTPException(status_code=404, detail="presentation not found")
    
    temp_dir = TEMP_FILE_SERVICE.create_temp_dir()

    async def inner():
        yield SSEStatusResponse(
            status="Generating presentation outlines ..."
        ).to_string()

        additional_context = ""
        if presentation.file_paths:
            documents_loader = DocumentsLoader(
                file_paths=presentation.file_paths
            )

            await documents_loader.load_documents(temp_dir=temp_dir)
            documents = documents_loader.documents 

            if documents:
                additional_context = "\n\n".join(documents)

        
        presentation_outlines_text = ""
        n_slides_to_generate = presentation.n_slides
        if presentation.include_table_of_contents:
            needed_toc_count = math.ceil((presentation.n_slides - 1) / 10)
            n_slides_to_generate -= math.ceil(
                (presentation.n_slides - needed_toc_count) / 10
            )    

        



    
    return StreamingResponse(inner(), media_type="text/event-stream")


        
