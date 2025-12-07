from fastapi import HTTPException
from openai import AsyncOpenAI

from utils.llm_provider import get_llm_provider, get_model
from enums.llm_provider import LLMProvider
from models.llm_message import LLMUserMessage, LLMSystemMessage, LLMMessage
from utils.get_env import (get_custom_llm_url_env, 
                           get_custom_llm_api_key_env,
                           get_qwen_llm_url_env,
                           get_qwen_llm_api_key_env,
                           
)

from services.llm_tool_calls_handler import LLMToolCallsHandler

from models.llm_tools import LLMDynamicTool, LLMTool
from typing import List, Optional
from models.llm_tool_call import OpenAIToolCall, OpenAIToolCallFunction
from models.llm_message import OpenAIAssistantMessage

class LLM_Client:
    def __init__(self):
        self.llm_provider = get_llm_provider()
        self._client = self._get_client()
        self.tool_call_handler = LLMToolCallsHandler(self)


    def _get_client(self):
        match self.llm_provider:
            case LLMProvider.CUSTOM:
                return self._get_custom_client()
            
            case LLMProvider.QWEN:
                return self._get_qwen_client()
            
            case _:
                raise HTTPException(
                    status_code=400,
                    detail="LLM Provider must be custom for now..",
                )
    

    def _get_custom_client(self):
        if not get_custom_llm_url_env():
            raise HTTPException(
                status_code=400,
                detail="Custom LLM not used"
            )
        
        return AsyncOpenAI(
            base_url=get_custom_llm_url_env(),
            api_key=get_custom_llm_api_key_env() or "null"
        )
    
    def _get_qwen_client(self):
        if not get_qwen_llm_url_env():
            raise HTTPException(
                status_code=400, 
                detail="Qwen LLM not used"
            )
        
        return AsyncOpenAI(
            base_url=get_qwen_llm_url_env(),
            api_key=get_qwen_llm_api_key_env() or "null"
        )
    

    async def _generate_openai(
            self,
            model:str,
            messages: List[LLMMessage]=None,
            max_tokens: Optional[int]=None,
            tools: Optional[List[dict]] = None,
            extra_body: Optional[dict]=None,
            depth: int = 0
    )-> str | None:

        client: AsyncOpenAI = self._client 
        response = await client.chat.completions.create(
            model = model,
            messages=[message.model_dump() for message in messages],
            max_completion_tokens=max_tokens,
            tools=tools,
            extra_body=extra_body
        )

        if len(response.choices) == 0:
            return None 
        
        tool_calls = response.choices[0].message.tool_calls
        
        if tool_calls:

            parsed_tool_calls = []
            for tool_call in tool_calls:
                parsed_tool_calls.append(
                    OpenAIToolCall(
                        id = tool_call.id,
                        type = tool_call.type, 
                        function = OpenAIToolCallFunction(
                            name = tool_call.function.name ,
                            arguments= tool_call.function.arguments
                        )
                    )
                )

            # prepare assistant messages 
            assistant_message = OpenAIAssistantMessage(
                    content=response.choices[0].message.content,
                    tool_calls = [tool_call.model_dump() for tool_call in parsed_tool_calls]
                )
            

            # prepare tool call messages
            tool_call_messages = await self.tool_call_handler.handle_tool_calls_openai(
                parsed_tool_calls
            )

            new_messages = [
                * messages,
                assistant_message,
                *tool_call_messages
            ]

            return await self._generate_openai(
                model = model, 
                messages = new_messages, 
                max_tokens=max_tokens,
                tools = tools,
                extra_body=extra_body,
                depth = depth + 1 
            )

        return response.choices[0].message.content

        
        
    # async def _generate_custom(self,
    #                            model: str, 
    #                            messages:List[LLMMessage],
    #                            max_tokens: Optional[int] = None,
    #                            depth: int = 0):
        
    #     # extra_body = {"enable_thinking": False} if self.disable_thinking() else None
        
    #     return await self._generate_openai(
    #         model=model,
    #         messages= messages,
    #         max_tokens=max_tokens,
    #         extra_body=extra_body,
    #         depth=body
    #     )
    
    async def _generate_qwen(
            self,
            model:str,
            messages: List[LLMMessage]=None,
            max_tokens: Optional[int]=None,
            tools: Optional[List[dict]] = None,
            extra_body: Optional[dict]=None,
            depth: int = 0
    ):
        client: AsyncOpenAI = self._client 
        response = await client.chat.completions.create(
            model = model,
            messages=[message.model_dump() for message in messages],
            max_completion_tokens=max_tokens,
            tools=tools,
            extra_body=extra_body
        )

        if len(response.choices) == 0:
            return None 
        
        tool_calls = response.choices[0].message.tool_calls

        return 


    
    async def generate(self, 
                       model: str, 
                       messages: List[LLMMessage],
                       max_tokens: Optional[int]= None, 
                       tools: Optional[List[type[LLMTool] | LLMDynamicTool]] = None, 
                       ):
        
        # match self.llm_provider:
        #     case LLMProvider.CUSTOM:
        #         content = await self._generate_custom(
        #             model=model, messages = messages, max_tokens=max_tokens 
        #         )
        #     case LLMProvider.OPENAI:
        #         content = await self._generate_openai(
        #             model=model, messages=messages, max_tokens=max_tokens
        #         )
        parsed_tools = self.tool_call_handler.parse_tools(tools)

        content = await self._generate_openai(
                model=model, messages=messages, max_tokens=max_tokens, 
                tools=parsed_tools
            )
        
        return content
        
  


    async def _get_system_prompt(self, messages: List[LLMMessage]):
        for msg in messages:
            if isinstance(msg, LLMSystemMessage):
                return msg.content
            
        return ""
    

    async def _search_qwen(self, query: str):
        # Qwen doesn't support OpenAI's /responses endpoint
        # You'll need to implement web search using a third-party API
        # For now, returning a placeholder message
        return f"Web search not implemented for Qwen. Query was: {query}"

    
    async def _search_openai(self, query: str):
        # https://platform.openai.com/docs/api-reference/responses/create?lang=python
        client : AsyncOpenAI = self._client
        response = await client.responses.create(
            model = get_model(),
            tools = [{"type" : "web_search_preview"}],
            input = query
        )

        print(response)
        return response.output_text

                



