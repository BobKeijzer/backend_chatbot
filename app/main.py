from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from .routes import router
import uuid
import os 
from dotenv import load_dotenv

load_dotenv

app = FastAPI(title="My AI Agent Backend")
FRONTEND_URL = os.getenv("FRONTEND_URL")

# -------------------------
# CORS setup
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Middleware: auto-set user_id cookie
# -------------------------
@app.middleware("http")
async def add_user_id_cookie(request: Request, call_next):
    response: Response = await call_next(request)
    
    # Check if user_id cookie exists
    if not request.cookies.get("user_id"):
        new_user_id = str(uuid.uuid4())
        response.set_cookie(
            key="user_id",
            value=new_user_id,
            max_age=60*60*24*365,  # 1 year
            path="/",
            httponly=True
        )
    return response

# -------------------------
# Include API routes
# -------------------------
app.include_router(router)
