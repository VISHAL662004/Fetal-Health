from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from groq_client import ask_groq
from knowledge_base import lookup_parameter

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_message = data.get("message", "")

    # 1. Check knowledge base first
    kb_answer = lookup_parameter(user_message.lower())
    if kb_answer:
        return {"response": kb_answer}

    # 2. Otherwise, ask Groq
    response = ask_groq(user_message)
    return {"response": response}
