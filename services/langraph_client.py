


from utils.llm_provider import get_llm_provider, get_model
from utils.get_env import (
    get_qwen_llm_url_env,
    get_qwen_llm_api_key_env,
    get_openai_llm_api_key_env,
    get_tool_calls_env
)
from enums.llm_provider import LLMProvider
from fastapi import HTTPException
from typing import List, Coroutine, Any, Callable, Optional
from models.llm_message import LLMMessage
from utils.parser import parse_bool_or_none
from utils.schema_utils import convert_schema_to_pydantic
from utils.dummy_functions import do_nothing_async
import dirtyjson

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool as langchain_tool



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

    def use_tool_call_for_structured_output(self)->bool:
        if self.llm_provider not in [LLMProvider.CUSTOM, LLMProvider.QWEN]:
            return False # No need to use a tool call because they support response format
        
        return parse_bool_or_none(get_tool_calls_env()) or False
    
    def _create_response_schema_dynamic_tool(self, response_schema: dict | type, strict: bool=False):
        """Create langchain tool for a response schema"""

        
        # Convert to Pydantic model (handles both dict and BaseModel)
        ResponseSchemaModel = convert_schema_to_pydantic(response_schema, strict)
        
        # Create the tool using the decorator
        @langchain_tool(args_schema=ResponseSchemaModel)
        def ResponseSchema(**kwargs) -> str:
            """Provide response to the user"""
            return "Response provided"
        
        return ResponseSchema
    

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
    
    async def generate_structured(
            self, 
            messages: List[LLMMessage],
            response_format: dict, 
            strict: bool = False,
            model: Optional[str] = None,
            max_tokens: Optional[int] = None 
    ) -> dict: 
        """Generate Strcuturd output from llm"""
        # convert llm message to langChain format 
        langchain_messages = []
        for msg in messages:
            msg_dict = msg.model_dump()
            langchain_messages.append({
                "role":msg_dict.get("role"),
                "content": msg_dict.get("content")
            })

        use_tool_call = self.use_tool_call_for_structured_output()
        if use_tool_call:
            # use the toll call trick for providers that don't support response_format
            response_schema_tool = self._create_response_schema_dynamic_tool(response_format, strict)

            # create an agent with the additional ReponseSchema tool 
            all_tools = [*self.tools, response_schema_tool]
            agent = create_react_agent(
                model = self.llm,
                tools = all_tools
            )

            result = await agent.ainvoke(
                {
                    "messages": langchain_messages
                }
            )


            # Extract the ResponseSchema tool call from messages
            for message in result["messages"]:
                if hasattr(message, "tool_calls") and message.tool_calls:
                    for tool_call in message.tool_calls:
                        if tool_call.get("name") == "ResponseSchema":
                            # extract the arguments (structured data)
                            args = tool_call.get("args", {})
                            return dict(args)

            # if no ResponseSchema tool was called, try parsing the content
            content = result["messages"][-1].content
            if content:
                return dict(dirtyjson.loads(content))

            raise HTTPException(
                status_code=400,
                detail="LLM didn't return structured content"
            ) 

        else:
            # use native strcutred output (with structued output)
            # this works for providers that support it (OpenAi, ..)
            structured_llm = self.llm.with_structured_output(
                response_format,
                method="json_schema" if strict else "json_mode"
            )

            # for structured output without agent, we can directly invoke
            result = await structured_llm.invoke(langchain_messages)

            return dict(result) if not isinstance(result, dict) else result
        


            




