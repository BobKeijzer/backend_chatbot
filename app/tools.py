import json
import sympy as sp
import re
import requests
from bs4 import BeautifulSoup
from langchain_community.tools import DuckDuckGoSearchResults

# -------------------
# Tool functions
# -------------------
def rag_tool(query: str, user_id: str, min_score: float = 0.3, k_amount: int = 5) -> str: 
    from .services import get_user_vectorstore
    vector_store = get_user_vectorstore(user_id)
    docs = vector_store.similarity_search_with_score(query, k=k_amount) # Add metadata filtering?
    print(f"tool called with {query}, {min_score}, {k_amount}")
    if len(docs) == 0:
        return "No docs uploaded."
    docs = [(d, score) for d, score in docs if score > min_score]
    sources = "\n\n".join(
        [
            f"File: {d.metadata.get('filename', '')}\n"
            f"Path: {d.metadata.get('folderpath', '')}\n"
            f"Content: {d.page_content}"
            for d, score in docs
        ]
    )
    return sources or "No relevant results found."

def search_tool(query: str) -> str:
    # Detect if input looks like a URL
    if query.startswith("http://") or query.startswith("https://"):
        try:
            response = requests.get(query, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            text = soup.get_text(separator=" ")
            text = re.sub(r"\s+", " ", text).strip()
            return text or "No readable content found on the page."
        except Exception as e:
            return f"Error fetching URL: {str(e)}"
    else:
        # Else treat input as a search query
        search = DuckDuckGoSearchResults(output_format="list")
        results = search.invoke(query)
        sources = "\n\n".join(
            [
                f"Title: {r.get('title', '')}\n"
                f"Link: {r.get('link', '')}\n"
                f"Content: {r.get('snippet', '')}"
                for r in results
            ]
        )
        return sources or "No results found."

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
        "description": "Search the user's uploaded documents for relevant information. Allows optional tuning with min_score and k_amount.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to run against uploaded documents."
                },
                "min_score": {
                    "type": "number",
                    "description": "Optional. Minimum similarity score threshold for results (default = 0.3).",
                    "default": 0.3
                },
                "k_amount": {
                    "type": "integer",
                    "description": "Optional. Number of top results to return (default = 5).",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    }
}


search_tool_schema = {
    "type": "function",
    "function": {
        "name": "search_web_online",
        "description": "Search the web for current information. Input can be a natural language query or a direct URL. If a URL is provided, the tool fetches and extracts the page content.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Either a natural language search query or a URL starting with http:// or https://"
                }
            },
            "required": ["query"]
        }
    }
}

calculator_tool_schema = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate complex mathematical expressions safely and fastly without needing intermediate steps.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A valid Sympy-compatible mathematical expression (e.g., 'sqrt(2) + 3**2', 'sin(pi/4)', '(67438 * 3435) / 7')."
                }
            },
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
