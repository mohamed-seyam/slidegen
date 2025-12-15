from openai import AsyncOpenAI


async def list_available_qwen_compatible_models(
        url: str, api_key: str
):
    client = AsyncOpenAI(
        base_url=url,
        api_key=api_key
    )

    models = (await client.models.list()).data
    if models:
        return list(map(lambda x: x.id, models))
    
    return []

