# 

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent import ask_agent  # 🔥 THIS connects everything

app = FastAPI()

# Allow Docusaurus frontend + Vercel frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # for local testing
        "https://physical-humanoids-hackathon-projec.vercel.app"  # ✅ your Vercel frontend
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
def chat(req: ChatRequest):
    answer = ask_agent(req.question)
    return {"answer": answer}