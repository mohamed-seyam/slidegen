from fastapi import HTTPException
from enums.llm_provider import LLMProvider
from utils.get_env import (
    get_llm_provider_env,
    get_custom_llm_api_key_env, 
    get_custom_image_api_url_env, 
    get_custom_model_env,
    get_qwen_model_env
)

def get_llm_provider():
    try:
        return LLMProvider(get_llm_provider_env())
    except:
        raise HTTPException(
            status_code = 500,
            details=f'invalid LLM Provider'
        )
    
def is_openai_selected():
    return get_llm_provider() == LLMProvider.OPENAI

def is_custom_selected():
    return get_llm_provider() == LLMProvider.CUSTOM

def is_qwen_selected():
    return get_llm_provider() == LLMProvider.QWEN


def get_model():
    selected_llm = get_llm_provider()

    if selected_llm == LLMProvider.CUSTOM:
        return get_custom_model_env()
    
    elif selected_llm == LLMProvider.QWEN:
        return get_qwen_model_env()
    
    else:
        raise HTTPException(
            status_code = 500,
            detail= "invalid llm provider"
        )
    

