from pydantic import BaseModel
from typing import Literal

class LLMToolCall(BaseModel):
    "Base Tool call class"
    pass 

class OpenAIToolCallFunction(BaseModel):
    name: str
    arguments: str


class OpenAIToolCall(LLMToolCall):
    id: str
    type: Literal["function"] = "function"
    function: OpenAIToolCallFunction
