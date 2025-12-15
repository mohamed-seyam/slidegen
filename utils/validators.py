from fastapi import UploadFile, HTTPException
from typing import List, Optional

def validate_files(
        field,
        nullable: bool,
        multiple: bool,
        max_size: int,
        accepted_types: List[str]

):
    if field:
        files: List[UploadFile] = field if multiple else [field]

        for each_file in files:
            if (max_size * 1024 * 1024) < each_file.size:
                raise HTTPException(
                    status_code=400,
                    detail=f"File {each_file.filename} exceeds the maximum allowed size of {max_size} MB."
                )
            elif each_file.content_type not in accepted_types:
                raise HTTPException(
                    status_code=400,
                    detail=f"File {each_file.filename} has an unsupported file type {each_file.content_type}."
                )
            
    elif not (field or nullable):
        raise HTTPException(
            status_code=400,
            detail="No files provided for upload."
        )