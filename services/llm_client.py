import dirtyjson
from fastapi import HTTPException
from openai import AsyncOpenAI
from typing import AsyncGenerator
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk as OpenAIChatCompletionChunk
)


from utils.llm_provider import get_llm_provider, get_model
from enums.llm_provider import LLMProvider
from models.llm_message import LLMUserMessage, LLMSystemMessage, LLMMessage
from utils.get_env import (get_custom_llm_url_env, 
                           get_custom_llm_api_key_env,
                           get_qwen_llm_url_env,
                           get_qwen_llm_api_key_env,
                           get_disable_thinking_env,
                           get_tool_calls_env,
                           
)
from utils.parser import parse_bool_or_none
from utils.schema_utils import ensure_strict_json_schema
from utils.dummy_functions import do_nothing_async

from services.llm_tool_calls_handler import LLMToolCallsHandler
from models.llm_tool_call import LLMToolCall

from models.llm_tools import LLMDynamicTool, LLMTool
from typing import List, Optional
from models.llm_tool_call import OpenAIToolCall, OpenAIToolCallFunction
from models.llm_message import OpenAIAssistantMessage

class LLM_Client:
    def __init__(self):
        self.llm_provider = get_llm_provider()
        self._client = self._get_client()
        self.tool_call_handler = LLMToolCallsHandler(self)

    # ? Use tool calls 
    def use_tool_calls_for_structured_output(self)-> bool:
        if self.llm_provider not in [LLMProvider.CUSTOM, LLMProvider.QWEN]:
            return False   # No need to use a tool call becuase they support response format
        
        return parse_bool_or_none(get_tool_calls_env()) or False
    

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
    
    def disable_thinking(self)-> bool:
        return parse_bool_or_none(get_disable_thinking_env()) or False

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
        return await self._generate_openai(
            model = model, 
            messages=messages,
            max_tokens=max_tokens,
            tools=tools,
            extra_body=extra_body, 
            depth = depth
        )


    
    async def generate(self, 
                       model: str, 
                       messages: List[LLMMessage],
                       max_tokens: Optional[int]= None, 
                       tools: Optional[List[type[LLMTool] | LLMDynamicTool]] = None, 
                       ):
  
        parsed_tools = self.tool_call_handler.parse_tools(tools)
        
        content = None 
        match self.llm_provider:
            case LLMProvider.OPENAI:
                content = await self._generate_openai(
                        model=model, 
                        messages=messages, 
                        max_tokens=max_tokens, 
                        tools=parsed_tools
                    )
            case LLMProvider.QWEN:
                content = await self._generate_qwen(
                    model = model,
                    messages = messages, 
                    max_tokens = max_tokens,
                    tools = parsed_tools
                )

        if content is None:
            raise HTTPException(
                status_code=400,
                detail="LLM didn't return any content"
            )
        return content
        
    

    # ? Generate Structured Content
    # async def _generate_openai_stuctured(
    #         self,
    #         model: str,
    #         messages: List[LLMMessage],
    #         response_format: dict,
    #         strict: bool = False,
    #         max_tokens: Optional[int] = None,
    #         tools: Optional[List[dict]] = None,
    #         extra_body: Optional[dict] = None,
    #         depth: int = 0
    # ) -> dict | None:
    #     client : AsyncOpenAI = self._client
     
    async def _generate_openai_structured(
        self,
        model: str,
        messages: List[LLMMessage],
        response_format: dict,
        strict: bool = False,
        max_tokens: Optional[int] = None,
        tools: Optional[List[dict]] = None,
        extra_body: Optional[dict] = None,
        depth: int = 0,
    ) -> dict | None:
        client: AsyncOpenAI = self._client
        response_schema = response_format
        all_tools = [*tools] if tools else None

        use_tool_calls_for_structured_output = (
            self.use_tool_calls_for_structured_output()
        )
        if strict and depth == 0:
            response_schema = ensure_strict_json_schema(
                response_schema,
                path=(),
                root=response_schema,
            )
        if use_tool_calls_for_structured_output and depth == 0:
            if all_tools is None:
                all_tools = []
            all_tools.append(
                self.tool_call_handler.parse_tool(
                    LLMDynamicTool(
                        name="ResponseSchema",
                        description="Provide response to the user",
                        parameters=response_schema,
                        handler=do_nothing_async,
                    ),
                    strict=strict,
                )
            )

        response = await client.chat.completions.create(
            model=model,
            messages=[message.model_dump() for message in messages],
            response_format=(
                {
                    "type": "json_schema",
                    "json_schema": (
                        {
                            "name": "ResponseSchema",
                            "strict": strict,
                            "schema": response_schema,
                        }
                    ),
                }
                if not use_tool_calls_for_structured_output
                else None
            ),
            max_completion_tokens=max_tokens,
            tools=all_tools,
            extra_body=extra_body,
        )

        if len(response.choices) == 0:
            return None

        content = response.choices[0].message.content

        tool_calls = response.choices[0].message.tool_calls
        has_response_schema = False

        if tool_calls:
            for tool_call in tool_calls:
                if tool_call.function.name == "ResponseSchema":
                    content = tool_call.function.arguments
                    has_response_schema = True

            if not has_response_schema:
                parsed_tool_calls = [
                    OpenAIToolCall(
                        id=tool_call.id,
                        type=tool_call.type,
                        function=OpenAIToolCallFunction(
                            name=tool_call.function.name,
                            arguments=tool_call.function.arguments,
                        ),
                    )
                    for tool_call in tool_calls
                ]
                tool_call_messages = (
                    await self.tool_calls_handler.handle_tool_calls_openai(
                        parsed_tool_calls
                    )
                )
                new_messages = [
                    *messages,
                    OpenAIAssistantMessage(
                        role="assistant",
                        content=response.choices[0].message.content,
                        tool_calls=[each.model_dump() for each in parsed_tool_calls],
                    ),
                    *tool_call_messages,
                ]
                content = await self._generate_openai_structured(
                    model=model,
                    messages=new_messages,
                    response_format=response_schema,
                    strict=strict,
                    max_tokens=max_tokens,
                    tools=all_tools,
                    extra_body=extra_body,
                    depth=depth + 1,
                )
        if content:
            if depth == 0:
                return dict(dirtyjson.loads(content))
            return content
        return None


    async def generate_structured(
            self,
            model: str,
            messages: List[LLMMessage],
            response_format: dict,
            strict: bool = False,
            tools: Optional[List[type[LLMTool] | LLMDynamicTool]] = None,
            max_tokens: Optional[int] = None,
        ) -> dict:
    
        parsed_tools = self.tool_call_handler.parse_tools(tools)

        content = None
        match self.llm_provider:
            case LLMProvider.OPENAI:
                content = await self._generate_openai_structured(
                    model = model, 
                    messages = messages,
                    response_format=response_format,
                    strict = strict,
                    tools=parsed_tools,
                    max_tokens = max_tokens
                )
            case LLMProvider.QWEN:
                content = await self._generate_openai_structured(
                    model = model, 
                    messages = messages,
                    response_format=response_format,
                    strict = strict,
                    tools=parsed_tools,
                    max_tokens = max_tokens
                )

        if content is None:
            raise HTTPException(
                status_code=400,
                detail= "LLM didn't return any content"
            )

        return content 

    async def _stream_openai(
            self,
            model: str,
            messages: List[LLMMessage],
            max_tokens: Optional[int] = None,
            tools: Optional[List[dict]] = None, 
            extra_body: Optional[dict] = None, 
            depth: int = 0 
    )-> AsyncGenerator:
        client: AsyncOpenAI = self._client

        tool_calls: List[LLMToolCall] = []
        current_index = 0
        current_id = None
        current_name = None
        current_arguments = None
        has_tool_calls = False  # Track if we've seen any tool calls
        in_tool_call_text = False  # Track if we're inside <tool_call> text
        tool_call_text_buffer = ""  # Accumulate <tool_call> text for manual parsing

        async for event in await client.chat.completions.create(
            model = model,
            messages = [msg.model_dump() for msg in messages],
            max_completion_tokens=max_tokens,
            tools= tools,
            extra_body=extra_body,
            stream=True
        ):
            event: OpenAIChatCompletionChunk = event
            if not event.choices:
                continue

            content_chunk = event.choices[0].delta.content
            tool_call_chunk = event.choices[0].delta.tool_calls

            if tool_call_chunk:
                has_tool_calls = True  # Mark that we've seen tool calls
                tool_index = tool_call_chunk[0].index
                tool_id = tool_call_chunk[0].id
                tool_name = tool_call_chunk[0].function.name
                tool_arguments = tool_call_chunk[0].function.arguments

                # Check if this is a new tool (index changed) or continuation of current tool
                if current_index != tool_index:
                    if current_id is not None:
                        tool_calls.append(
                            OpenAIToolCall(
                                id=current_id,
                                type="function",
                                function=OpenAIToolCallFunction(
                                    name=current_name,
                                    arguments=current_arguments
                                )
                            )
                        )

                    current_index = tool_index
                    current_id = tool_id
                    current_name = tool_name
                    current_arguments = tool_arguments

                else:
                    current_name = tool_name or current_name
                    current_id = tool_id or current_id
                    if current_arguments is None:
                        current_arguments = tool_arguments
                    elif tool_arguments:
                        current_arguments += tool_arguments

            # Handle content chunks
            if content_chunk and not has_tool_calls:
                # Check if we're entering a tool call text block
                if '<tool_call>' in content_chunk:
                    in_tool_call_text = True
                    tool_call_text_buffer = content_chunk  # Start accumulating

                # Accumulate tool call text
                elif in_tool_call_text:
                    tool_call_text_buffer += content_chunk

                # Check if we're exiting a tool call text block
                if '</tool_call>' in content_chunk:
                    in_tool_call_text = False
                    continue  # Skip this chunk entirely

                # Only yield if we're not inside a tool call text block
                if not in_tool_call_text:
                    yield content_chunk

        # After stream ends, save the last tool call if any
        if current_id is not None:
            tool_calls.append(
                OpenAIToolCall(
                    id=current_id,
                    type="function",
                    function=OpenAIToolCallFunction(
                        name = current_name,
                        arguments=current_arguments
                    )
                )
            )

        # Parse tool call text if we accumulated any (Qwen fallback)
        if tool_call_text_buffer and not tool_calls:
            import json
            import re
            # Extract JSON from <tool_call>...</tool_call>
            match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', tool_call_text_buffer, re.DOTALL)
            if match:
                try:
                    tool_data = json.loads(match.group(1))
                    tool_calls.append(
                        OpenAIToolCall(
                            id="manual_tool_call_0",
                            type="function",
                            function=OpenAIToolCallFunction(
                                name=tool_data.get("name"),
                                arguments=json.dumps(tool_data.get("arguments", {}))
                            )
                        )
                    )
                    has_tool_calls = True
                except json.JSONDecodeError:
                    pass  # Silently fail if parsing fails

        if tool_calls:
            tool_call_messages = await self.tool_call_handler.handle_tool_calls_openai(
                tool_calls
            )

            new_messages = [
                *messages,
                OpenAIAssistantMessage(
                    role = "assistant",
                    content=None,
                    tool_calls=[each.model_dump() for each in tool_calls]

                ),
                *tool_call_messages
            ]

            async for event in self._stream_openai(
                model = model,
                messages = new_messages,
                max_tokens=max_tokens,
                tools = tools,
                extra_body=extra_body,
                depth = depth + 1
            ):
                yield event 
            
    
    def stream(
            self,
            model:str,
            messages: List[LLMMessage],
            max_tokens: Optional[int] = None,
            tools: Optional[List[type[LLMTool] | LLMDynamicTool]] = None,
    ):
        
        parsed_tools = self.tool_call_handler.parse_tools(tools) if tools else None 

        # stream based on provider
        match self.llm_provider:
            case  LLMProvider.OPENAI:
                return self._stream_openai(
                    model = model, 
                    messages=messages,
                    max_tokens=max_tokens,
                    tools = parsed_tools
                )
            
            case LLMProvider.QWEN:
                extra_body = {"enable_thinking": False} if self.disable_thinking() else None
                return self._stream_openai(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    tools=parsed_tools,
                    extra_body=extra_body
                )
            case _:
                raise HTTPException(
                    status_code=400,
                    detail=f"Streaming not implemented for provider: {self.llm_provider}"
                )


        
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

                



