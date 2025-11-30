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

def get_image_provider_env():
    return os.getenv("IMAGE_PROVIDER", "DALL-E")

def get_custom_image_api_url_env():
    return os.getenv("CUSTOM_IMAGE_API_URL")

def get_custom_image_api_key_env():
    return os.getenv("CUSTOM_IMAGE_API_KEY")

def get_database_url_env():
    return os.getenv("DATABASE_URL")
