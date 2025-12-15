import os 

def get_app_data_directory_env():
    return os.getenv("APP_DATA_DIRECTORY", "./data")

def get_llm_provider_env():
    return os.getenv("LLM", "custom")

def get_custom_llm_url_env():
    return os.getenv("CUSTOM_LLM_URL")

def get_custom_llm_api_key_env():
    return os.getenv("CUSTOM_LLM_API_KEY")

def get_custom_model_env():
    return os.getenv("CUSTOM_LLM_MODEL")


def get_qwen_llm_url_env():
    return os.getenv("QWEN_LLM_URL")

def get_qwen_llm_api_key_env():
    return os.getenv("QWEN_LLM_API_KEY")

def get_qwen_model_env():
    return os.getenv("QWEN_LLM_MODEL")


def get_openai_llm_url_env():
    return os.getenv("OPENAI_LLM_URL")

def get_openai_llm_api_key_env():
    return os.getenv("OPENAI_LLM_API_KEY")


def get_image_provider_env():
    return os.getenv("IMAGE_PROVIDER", "DALL-E")

def get_custom_image_api_url_env():
    return os.getenv("CUSTOM_IMAGE_API_URL")

def get_custom_image_api_key_env():
    return os.getenv("CUSTOM_IMAGE_API_KEY")

def get_database_url_env():
    return os.getenv("DATABASE_URL")


def get_disable_thinking_env():
    return os.getenv("DISABLE_THINKING")

def get_tool_calls_env():
    return os.getenv("TOOL_CALLS")


def get_temp_directory_env():
    return os.getenv("TEMP_DIRECTORY")
