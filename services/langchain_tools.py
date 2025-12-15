from langchain_core.tools import tool
from datetime import datetime


@tool
def multiply(x: float, y: float) -> str:
    """Multiply two numbers together"""
    return str(x * y)


@tool
def get_current_datetime() -> str:
    """Get the current date and time"""
    current_time = datetime.now()
    return f"{current_time.strftime('%A, %B %d, %Y')} at {current_time.strftime('%I:%M:%S %p')}"


@tool
def search_web(query: str) -> str:
    """Search the web for information"""
    # TODO: Implement with SerpAPI, DuckDuckGo, or other search API
    return f"Web search not implemented. Query was: {query}"


# Export all tools as a list for easy import
ALL_TOOLS = [
    multiply,
    get_current_datetime,
    search_web
]
