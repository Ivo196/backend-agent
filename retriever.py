from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool
from config import PINECONE_INDEX, EMBEDDING_MODEL

def get_retriever():
    pinecone = Pinecone()
    index = pinecone.Index(PINECONE_INDEX)

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL
        )

    vectorStore = PineconeVectorStore(
        index,
        embeddings
        )
    
    retrieve = vectorStore.as_retriever(search_kwargs={"k": 2})
    
    return retrieve