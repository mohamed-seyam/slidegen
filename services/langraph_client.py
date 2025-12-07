


from utils.llm_provider import get_llm_provider, get_model
from utils.get_env import (
    get_qwen_llm_url_env, 
    get_qwen_llm_api_key_env,
    get_openai_llm_api_key_env
)
from enums.llm_provider import LLMProvider
from fastapi import HTTPException
from typing import List, Coroutine, Any, Callable, Optional
from models.llm_message import LLMMessage

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent



class Langraph_LLM_Client:
    def __init__(self, tools: List[Callable[..., Coroutine[Any, Any, str]]]):
        self.llm_provider = get_llm_provider()
        self.llm = self._get_llm()
        self.llm_model_name = get_model()

        self.tools = tools

        # Agent will be created per request if model/max_tokens need to be customized
        self.agent = create_react_agent(
            model = self.llm,
            tools = self.tools
 
        )


    def _get_llm(self):
        match self.llm_provider:
            case LLMProvider.QWEN:
                if not get_qwen_llm_url_env():
                    raise HTTPException(
                        status_code=400,
                        detail="Qwen LLM Url not configured"
                    )

                return ChatOpenAI(
                    model = get_model(),
                    base_url = get_qwen_llm_url_env(),
                    api_key = get_qwen_llm_api_key_env(),
                )

            case LLMProvider.OPENAI:
                if not get_openai_llm_api_key_env():
                    raise HTTPException(
                        status_code=400,
                        detail="API key must be provided"
                    )

                return ChatOpenAI(api_key=get_openai_llm_api_key_env())
            
            
            case _:
                raise HTTPException(
                        status_code=400,
                        detail=f"Un supported LLM provider: {self.llm_provider}"
                    )
            

    async def generate(
            self,
            messages: List[LLMMessage],
            model: Optional[str] = None,
            max_tokens: Optional[int] = None
    ):
        # Convert LLM Message to LangChain format
        langchain_messages = []
        for msg in messages:
            msg_dict = msg.model_dump()
            langchain_messages.append(
                {
                    "role" : msg_dict.get("role"),
                    "content" : msg_dict.get("content")
                }
            )

        
        # Use default agent
        result = await self.agent.ainvoke(
            {
                "messages": langchain_messages
            }
        )

        return result["messages"][-1].content