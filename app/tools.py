import json
import sympy as sp

# -------------------
# Tool functions
# -------------------
def rag_tool(query: str, user_id: str) -> str:
    from .services import get_user_vectorstore
    vector_store = get_user_vectorstore(user_id)
    docs = vector_store.similarity_search(query, k=4)
    if len(docs) == 0:
        return "No docs uploaded."
    sources = "\n\n".join(
        [
            f"File: {d.metadata.get('filename', '')}\n"
            f"Path: {d.metadata.get('folderpath', '')}\n"
            f"Content: {d.page_content}"
            for d in docs
        ]
    )
    return sources or "No relevant results found."

def search_tool(query: str) -> str:
    from langchain_community.tools import DuckDuckGoSearchResults
    search = DuckDuckGoSearchResults()
    results = search.invoke(query)
    return results or "No results found."

def calculator_tool(expression: str) -> str:
    try:
        result = sp.sympify(expression).evalf()
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"


# -------------------
# Tool schemas
# -------------------
rag_tool_schema = {
    "type": "function",
    "function": {
        "name": "search_uploaded_files",
        "description": "Search the user's uploaded documents for relevant information.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"]
        }
    }
}

search_tool_schema = {
    "type": "function",
    "function": {
        "name": "search_web_online",
        "description": "Search the web for current information.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"]
        }
    }
}

calculator_tool_schema = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate a mathematical expression safely and return the result.",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"]
        }
    }
}

# -------------------
# General tool logic
# -------------------

TOOL_MAPPING = {
    "search_uploaded_files": rag_tool,
    "search_web_online": search_tool,
    "calculator": calculator_tool,
}

TOOLS = [rag_tool_schema, search_tool_schema, calculator_tool_schema]


def get_tool_responses(response, user_id: str):
    tool_calls = response.choices[0].message.tool_calls or []
    tool_messages = []

    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)

        if tool_name == "search_uploaded_files":
            tool_result = TOOL_MAPPING[tool_name](user_id=user_id, **tool_args)
        else:
            tool_result = TOOL_MAPPING[tool_name](**tool_args)

        tool_messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": tool_result,
        })

    return tool_messages
