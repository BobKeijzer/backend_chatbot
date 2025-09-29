import os
import sqlite3
import json
from datetime import datetime, timezone
from dotenv import load_dotenv
from typing import List
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss
from .tools import TOOLS, get_tool_responses
from langchain_core.documents import Document
from openai import (
    OpenAI,
    RateLimitError,
    APIConnectionError,
    APIError,
)
import random
from collections import defaultdict
from typing import Dict, Any
import uuid
import httpx

load_dotenv()

# -------------------------
# Constants
# -------------------------
DB_PATH = "chats.db"
INDEX_DIR = "user_indexes"
conn = sqlite3.connect(DB_PATH)
conn.execute(
    """CREATE TABLE IF NOT EXISTS chats (
        chat_id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        name TEXT DEFAULT 'New Chat',
        messages TEXT,
        timestamp TEXT
    )"""
)
conn.commit()
conn.close()

OPENROUTER_MODEL = "deepseek/deepseek-chat-v3.1:free"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

EMBED_MODEL = "all-MiniLM-L6-v2"
DIMENSION = 384

with open("system.txt") as f:
    SYSTEM_MESSAGE = f.read()

# Database helper
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# -------------------------
# Chat history
# -------------------------
def get_user_chats(user_id: str):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT chat_id, name, timestamp FROM chats WHERE user_id=? ORDER BY timestamp DESC",
        (user_id,)
    )
    rows = cur.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_messages(chat_id: str):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT messages FROM chats WHERE chat_id=?", (chat_id,))
    row = cur.fetchone()
    conn.close()
    if row:
        return json.loads(row[0])
    return [{"role": "system", "content": SYSTEM_MESSAGE}]

def save_messages(user_id: str, chat_id: str, messages: List[dict]):
    conn = get_db_connection()
    cur = conn.cursor()
    timestamp = datetime.now(timezone.utc).isoformat()
    cur.execute(
        "INSERT OR REPLACE INTO chats (chat_id, user_id, messages, timestamp) VALUES (?, ?, ?, ?)",
        (chat_id, user_id, json.dumps(messages), timestamp)
    )
    conn.commit()
    conn.close()

# -------------------------
# FAISS handling per user
# -------------------------
def get_user_vectorstore(user_id: str) -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    index_folder = os.path.join(INDEX_DIR, f"faiss_user_{user_id}.index")
    os.makedirs(INDEX_DIR, exist_ok=True)

    if os.path.exists(index_folder):
        return FAISS.load_local(
            index_folder,
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        return FAISS(
            embedding_function=embeddings,
            index=faiss.IndexFlatIP(DIMENSION),
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

def save_user_vectorstore(user_id: str, vectorstore: FAISS):
    os.makedirs(INDEX_DIR, exist_ok=True) 
    index_folder = os.path.join(INDEX_DIR, f"faiss_user_{user_id}.index")
    vectorstore.save_local(index_folder)

# -------------------------
# File handling
# -------------------------
def add_files(user_id: str, documents: List[Document]):
    vectorstore = get_user_vectorstore(user_id)
    vectorstore.add_documents(documents)
    save_user_vectorstore(user_id, vectorstore)

def clear_user_rag(user_id: str):
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = FAISS(
        embedding_function=embeddings,
        index=faiss.IndexFlatIP(DIMENSION),
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    save_user_vectorstore(user_id, vectorstore)
    return vectorstore

def get_uploaded_files_summary(user_id: str) -> str:
    vectorstore = get_user_vectorstore(user_id)
    if not vectorstore:
        return ""

    docs = [doc for _, doc in vectorstore.docstore._dict.items()]
    seen_ids = set()
    files_summary_list = []

    for doc in docs:
        meta = doc.metadata
        file_id = meta.get("id", "")
        if file_id in seen_ids:
            continue  # skip duplicate chunks
        seen_ids.add(file_id)

        f = {
            "name": meta.get("filename", ""),
            "folderPath": meta.get("folderpath", "")
        }
        files_summary_list.append(f)

    files_summary_list.sort(key=lambda x: (x["folderPath"], x["name"]))
    lines = [
        f"- {f['name']} (folder: {f['folderPath'] if f['folderPath'] else 'Top-level'})"
        for f in files_summary_list
    ]
    return "\n".join(lines)

def get_user_file_metadata(user_id: str) -> Dict[str, Any]:
    vectorstore = get_user_vectorstore(user_id)
    if not vectorstore:
        return {"files": [], "folders": []}

    docs = [doc for _, doc in vectorstore.docstore._dict.items()]

    files_map = {}  
    folders_map = defaultdict(list)  
    seen_ids = set()  

    for doc in docs:
        meta = doc.metadata
        file_id = meta.get("id", "")
        if file_id in seen_ids:
            continue  # skip duplicate chunks
        seen_ids.add(file_id)

        uploaded_file = {
            "id": file_id,
            "name": meta.get("filename", ""),
            "folderPath": meta.get("folderpath", ""),
            "file": None,  # raw File objects are not returned
        }

        if uploaded_file["folderPath"]:
            root_folder = uploaded_file["folderPath"].split("/")[0]
            folders_map[root_folder].append(uploaded_file)
        else:
            files_map[file_id] = uploaded_file

    folders = [
        {
            "id": str(uuid.uuid4()),
            "name": folder_name,
            "files": files_list
        }
        for folder_name, files_list in folders_map.items()
    ]
    files = list(files_map.values())

    return {"files": files, "folders": folders}



# -------------------------
# LLM interaction
# -------------------------
def call_llm(user_message: str, chat_id: str, user_id: str) -> list[dict]:
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )

    messages = get_messages(chat_id)

    # Update system message with file metadata and current date for tool call context
    files_summary = get_uploaded_files_summary(user_id)
    timestamp = datetime.now(timezone.utc).isoformat()
    messages[0]["content"] = SYSTEM_MESSAGE + "\n\nUploaded Document MetaData:\n" + files_summary + "\n\nCurrent Date: " + timestamp 

    messages.append({"role": "user", "content": user_message})
    save_messages(user_id, chat_id, messages)

    max_iterations = 10
    iteration_count = 0
    new_messages = []

    # Helper to add errors as assistant responses
    def _error_message(text: str) -> dict:
        error_message = {"role": "assistant", "content": text}
        messages.append(error_message)
        save_messages(user_id, chat_id, messages)
        new_messages.append(error_message)
        return error_message

    try:
        while iteration_count < max_iterations:
            iteration_count += 1

            resp = client.chat.completions.create(
                model=OPENROUTER_MODEL,
                tools=TOOLS,
                messages=messages,
            )

            msg = resp.choices[0].message
            msg_dict = msg.to_dict()
            messages.append(msg_dict)
            save_messages(user_id, chat_id, messages)
            new_messages.append(msg_dict)

            if msg.tool_calls:
                tool_msgs = get_tool_responses(resp, user_id)
                for tool_msg in tool_msgs:
                    messages.append(tool_msg)
                    save_messages(user_id, chat_id, messages)
                    new_messages.append(tool_msg)
            else:
                break
        if iteration_count >= 10: 
            _error_message("I had to stop because too many tool calls were triggered in a row. Try rephrasing your request.")

    except RateLimitError:
        _error_message("Rate limit exceeded. Please try again later or upgrade your plan.")
    except APIConnectionError:
        _error_message("Network error. Unable to reach the AI service. Please check your connection.")
    except httpx.TimeoutException:
        _error_message("The request timed out. Please try again.")
    except APIError:
        _error_message("The AI service returned an error. Please try again later.")
    except Exception as e:
        _error_message(f"Unexpected error: {str(e)}")

    return new_messages



