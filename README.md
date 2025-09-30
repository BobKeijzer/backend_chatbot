# Backend Chatbot - PAI (Personal AI)

This is the **backend service** for the WebAI project, built with [FastAPI](https://fastapi.tiangolo.com/).  
It powers a separate **Next.js frontend** and provides APIs for:

- Managing chat sessions  
- Uploading and processing documents  
- Running **RAG (Retrieval-Augmented Generation)** pipelines with [LangChain](https://www.langchain.com/)  
- Using **agent tools** to extend LLM capabilities  

The project was created for **learning purposes**, focusing on FastAPI, LangChain, RAG, FAISS, Agents with tools and cloud deployment.

---

## Features

- SQLite chat persistence  
- File upload & RAG indexing with FAISS + HuggingFace embeddings  
- LLM interaction via OpenRouter API  
- Tool calling support for agent behavior  
- RESTful APIs consumed by a Next.js frontend  

---

## Frontend

Frontend lives in a separate repo:  
[frontend_chatbot](https://github.com/BobKeijzer/frontend_chatbot)  

Hosted on **Vercel**, while the backend is designed to run in a Docker container or just as code on **Azure App Services**.

---

## Getting Started

### Prerequisites
- Python 3.13  
- [pip](https://pip.pypa.io/)  
- Docker (optional, for containerization)

### Installation
```bash
git clone https://github.com/BobKeijzer/backend_chatbot.git
cd backend_chatbot

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Environment Variables

Configure a `.env` (ignored by git):

```env
OPENROUTER_API_KEY=your_key_here
FRONTEND_URL=http://localhost:3000
```

In [Azure](https://azure.microsoft.com) App Services, set these as **Application Settings** instead of using `.env`.

### Run locally

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Backend available at [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Docker

```bash
# Build image
docker build -t backend_chatbot .

# Run container
docker run -d -p 8000:8000 backend_chatbot
```

---

## Deployment Plans

* **Frontend** → Vercel (Next.js)
* **Backend** → Azure App Services (Docker container or code), connect via [Docker Hub](https://hub.docker.com) when using Docker

---

## Learning Goals

* FastAPI for production-ready APIs
* LangChain RAG pipelines with FAISS
* Tool-augmented LLM workflows
* Azure deployment flows (with Docker)

---

## License

MIT – free for learning and experimentation 

