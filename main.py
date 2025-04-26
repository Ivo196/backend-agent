from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent import process_chat 
from typing import Optional

class Post(BaseModel):
    message: str
    id: Optional[str] = None
    
app = FastAPI(title="Max's API", description="A simple API for Max's projects")

@app.get("/")
def read_root():
    return {"message": "Hello Ivo"}


@app.post("/chat")
def chat(post: Post):
    user_input = post.message
    id = post.id
    print(user_input, id)
    response = process_chat(user_input, id)
    print(response)
    return {"message": response}

