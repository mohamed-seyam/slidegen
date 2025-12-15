import uuid
from typing import Annotated, Optional, List
from fastapi import APIRouter, Body, Depends
from models.sql.presentation import PresentationModel
from enums.tone import Tone 
from enums.verbosity import Verbosity
from sqlalchemy.ext.asyncio import AsyncSession
from services.database import get_async_session
from fastapi import HTTPException


PRESENTATION_ROUTER = APIRouter(
    prefix="/presentation",
    tags=["presentation"]
)

@PRESENTATION_ROUTER.post("/create", response_model=PresentationModel)
async def create_presentation(
    content: Annotated[str, Body()],
    n_slides: Annotated[int, Body()],
    language: Annotated[str, Body()],
    file_paths: Annotated[Optional[List[str]], Body()] = None,
    tone: Annotated[Tone, Body()] = Tone.DEFAULT,
    verbosity: Annotated[Verbosity, Body()] = Verbosity.STANDARD,
    instructions: Annotated[Optional[str], Body()] = None,
    include_table_of_contents :Annotated[bool, Body()] = False,
    include_title_slide: Annotated[bool, Body()] = True,
    web_search : Annotated[bool, Body()] = False,
    sql_session: AsyncSession = Depends(get_async_session)
):
    if include_table_of_contents and n_slides < 3:
        raise HTTPException(
            status_code=400,
            detail="Number of slides cannot be less than 3 if table of contents is included",
        )
    
    presentation_id = uuid.uuid4()

    presentation = PresentationModel(
        id=presentation_id,
        content=content,
        n_slides=n_slides,
        language=language,
        file_paths=file_paths,
        tone=tone.value,
        verbosity=verbosity.value,
        instructions=instructions,
        include_table_of_contents=include_table_of_contents,
        include_title_slide=include_title_slide,
        web_search=web_search,
    )

    sql_session.add(presentation)
    await sql_session.commit()

    return presentation
