from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent import process_chat, agentExecutor 
from langchain_core.messages import HumanMessage, AIMessage
class Post(BaseModel):
    message: str
    
chat_history: list = []

app = FastAPI(title="Max's API", description="A simple API for Max's projects")

@app.get("/")
def read_root():
    return {"message": "Hello Ivo"}


@app.post("/chat")
def chat(post: Post):
    user_input = post.message
    print(user_input)
    response = process_chat(agentExecutor, user_input, chat_history)
    print(response)
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response))
    print(chat_history)
    return {"message": response}

