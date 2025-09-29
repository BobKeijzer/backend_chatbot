from fastapi import APIRouter, Request, Cookie, UploadFile, File, HTTPException, Form, Depends
from typing import List
from uuid import uuid4
import sqlite3, json
from .services import (
    save_messages,
    get_messages,
    get_user_chats,
    add_files,
    get_user_file_metadata,
    clear_user_rag,
    call_llm
)
from langchain_core.documents import Document
from .rag import chunk_file
from .services import DB_PATH, SYSTEM_MESSAGE
from typing import Literal, Optional
from pydantic import BaseModel

router = APIRouter()

# -------------------------
# Pydantic models
# -------------------------
class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system", "tool"]
    content: str
    tool_call_id: Optional[str] = None

# Dependency
def get_user_id(user_id: str = Cookie(None)):
    if not user_id:
        raise HTTPException(status_code=400, detail="Missing user_id cookie")
    return user_id

# Check backend health
@router.get("/health")
async def health():
    return {"status": "ok"}

# -------------------------
# Chat endpoints
# -------------------------
@router.post("/chat/message/{chat_id}")
def send_message(message: ChatMessage, chat_id: str, user_id: str = Depends(get_user_id)):
    user_message = message.content
    messages = call_llm(user_message, chat_id, user_id)
    return messages

@router.post("/chat/new")
def new_chat(user_id: str = Depends(get_user_id)):
    chat_id = str(uuid4())
    save_messages(user_id, chat_id, [{"role": "system", "content": SYSTEM_MESSAGE}])
    return {"chat_id": chat_id}

@router.get("/chat/load/{chat_id}")
def load_chat(chat_id: str, user_id: str = Depends(get_user_id)):
    messages = get_messages(chat_id)
    visible_messages = [m for m in messages if m["role"] in ("user", "assistant", "tool")]
    return {"chat_id": chat_id, "messages": visible_messages}

@router.get("/chat/list")
def list_chats(user_id: str = Depends(get_user_id)):
    return get_user_chats(user_id)

@router.delete("/chat/delete/{chat_id}")
def delete_chat(chat_id: str, user_id: str = Depends(get_user_id)):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM chats WHERE chat_id=? AND user_id=?", (chat_id, user_id))
    conn.commit()
    conn.close()
    return {"status": "ok"}

@router.post("/chat/rename/{chat_id}")
def rename_chat(chat_id: str, new_name: str = Form(...), user_id: str = Depends(get_user_id)):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("UPDATE chats SET name=? WHERE chat_id=? AND user_id=?", (new_name, chat_id, user_id))
    conn.commit()
    conn.close()
    return {"status": "ok"}

# -------------------------
# File endpoints
# -------------------------
@router.post("/files/upload")
async def upload_files(request: Request, files: List[UploadFile] = File(...), user_id: str = Depends(get_user_id)):
    docs = []
    form = await request.form() 
    metadata_list = form.getlist("metadata")
    for file , meta_json in zip(files, metadata_list):
        chunks = chunk_file(file.file, file.filename)
        meta = json.loads(meta_json)
        for c in chunks:
            docs.append(Document( 
                page_content=c, 
                metadata={ 
                    "filename": meta["name"], 
                    "folderpath": meta["folderPath"], 
                    "id": meta["id"] } 
                )) 
    add_files(user_id, docs)
    return {"status": "ok", "added": len(docs)}

@router.get("/files/metadata")
def list_files(user_id: str = Cookie(None)):
    if not user_id:
        raise HTTPException(status_code=400, detail="Missing user_id cookie")
    return get_user_file_metadata(user_id)
    

@router.post("/files/clear_rag")
def clear_rag(user_id: str = Depends(get_user_id)):
    clear_user_rag(user_id)
    return {"status": "ok"}
