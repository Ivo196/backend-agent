from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent import process_chat, agentExecutor 
class Post(BaseModel):
    message: str
    
app = FastAPI(title="Max's API", description="A simple API for Max's projects")

@app.get("/")
def read_root():
    return {"message": "Hello Ivo"}


@app.post("/chat")
def chat(post: Post):
    user_input = post.message
    print(user_input)
    response = process_chat(agentExecutor, user_input)
    print(response)
    return {"message": response}

