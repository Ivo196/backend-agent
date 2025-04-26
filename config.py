# config.py
import os
from dotenv import load_dotenv
   
load_dotenv()

PINECONE_INDEX = "langchain"
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

UPSTASH_URL = os.getenv("UPSTASH_REDIS_URL")
UPSTASH_TOKEN = os.getenv("UPSTASH_REDIS_TOKEN")