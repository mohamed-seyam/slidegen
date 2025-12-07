from pydantic import BaseModel
from typing import Literal, Optional, List 


class LLMMessage(BaseModel):
    pass 


class LLMUserMessage(LLMMessage):
    role: Literal["user"] = "user"
    content: str 


class LLMSystemMessage(LLMMessage):
    role : Literal["system"] = "system"
    content: str


class OpenAIToolCallMessage(LLMMessage):
    # https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/function-calling?view=foundry-classic&tabs=python-secure
    # https://learn.microsoft.com/en-us/answers/questions/1726523/missing-parameter-tool-call-id-messages-with-role
    role: Literal["tool"] = "tool" 
    tool_call_id: str 
    content : str 

class OpenAIAssistantMessage(LLMMessage):
    role: Literal["assistant"] = "assistant"
    content: str | None = None 
    tool_calls: Optional[List[dict]] = None