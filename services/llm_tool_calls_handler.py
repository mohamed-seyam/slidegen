
import asyncio
from datetime import datetime
from typing import Any, Callable, Coroutine, List, Optional
from enums.llm_provider import LLMProvider 
from models.llm_tools import SearchWebTool, MultiplyTool
from models.llm_tools import LLMTool, LLMDynamicTool
from utils.schema_utils import ensure_strict_json_schema
from models.llm_tool_call import OpenAIToolCall
from models.llm_message import OpenAIToolCallMessage
from fastapi import HTTPException

class LLMToolCallsHandler:
    def __init__(self, client):
        from .llm_client import LLM_Client
        self.client : LLM_Client = client

        self.tools_map: dict[str, Callable[..., Coroutine[Any, Any, str]]] = {
            "SearchWebTool" : self.search_web_tool_handler,
            "GetCurrentDataTimeTool" : self.get_current_datetime_tool_call_handler,
            "MultiplyTool" : self.multiply
        }

        self.dynamic_tools : List[LLMDynamicTool] = []

    def parse_tools(self, tools: Optional[List[type[LLMTool] | LLMDynamicTool]] = None):
        if tools is None:
            return None 

        parsed_tools = map(self.parse_tool, tools)
        return list(parsed_tools) 

    def parse_tool(self, tool: type[LLMTool] | LLMDynamicTool, strict : bool = False):
        if isinstance(tool, LLMDynamicTool):
            self.dynamic_tools.append(tool)
        
        match self.client.llm_provider:
            # case LLMProvider.OPENAI:
            #     self.parse_tool_openai()
            
            case LLMProvider.QWEN:
                return self.parse_tool_qwen(tool, strict) 

            case _ :
                raise ValueError(
                    "LLM provider must be qwen"
                )

    def parse_tool_qwen(self, tool: type[LLMTool] | LLMDynamicTool, strict : bool = False):
        if isinstance(tool, LLMDynamicTool):
            name = tool.name 
            description = tool.description
            parameters = tool.parameters

        else:
            name = tool.__name__ 
            description = tool.__doc__ or ""
            parameters = tool.model_json_schema()

        if strict: 
            parameters = ensure_strict_json_schema(parameters, path=(), root=parameters)

        return {
            "type" : "function",
            "function": {
                "name" : name,
                "description" : description,
                "strict" : strict,
                "parameters" : parameters
            },
        }
    
    async def handle_tool_calls_openai(self, tool_calls: List[OpenAIToolCall]) -> OpenAIToolCallMessage:
        async_tool_calls_tasks = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            tool_handler = self.get_tool_handler(tool_name)
            async_tool_calls_tasks.append(tool_handler(tool_call.function.arguments))

        tool_call_results: List[str] = await asyncio.gather(*async_tool_calls_tasks)
        tool_call_messages = [
            OpenAIToolCallMessage(
                content=result,
                tool_call_id=tool_call.id,
            )
            for tool_call, result in zip(tool_calls, tool_call_results)
        ]
        return tool_call_messages


    def get_tool_handler(self, tool_name: str) -> Callable[..., Coroutine[Any, Any, str]]:
        handler = self.tools_map.get(tool_name)
        if handler:
            return handler 
        
        else:
            dynamic_tools = list(
                filter(lambda tool: tool.name == tool_name, self.dynamic_tools)
            )

            if dynamic_tools:
                return dynamic_tools[0].handler
            
        raise HTTPException(status_code=500, detail=f"Tool {tool_name} not found")
    

    async def search_web_tool_handler(self, arguments: str):
        match self.client.llm_provider:
            case LLMProvider.QWEN:
                return await self.search_web_tool_call_handler_qwen(arguments)

            case LLMProvider.OPENAI:
                return await self.search_web_tool_call_handler_openai(arguments)


    async def search_web_tool_call_handler_qwen(self, arguments: str) -> str:
        args = SearchWebTool.model_validate_json(arguments)
        return await self.client._search_qwen(args.query)
    
    async def search_web_tool_call_handler_openai(self, arguments: str) -> str:
        args = SearchWebTool.model_validate_json(arguments)
        return await self.client._search_openai(args.query)


    async def get_current_datetime_tool_call_handler(self, _) -> str:
        current_time = datetime.now()
        return f"{current_time.strftime('%A, %B %d, %Y')} at {current_time.strftime('%I:%M:%S %p')}"


    async def multiply(self, arguments: str): 
        args = MultiplyTool.model_validate_json(arguments)
        return str(args.x * args.y)