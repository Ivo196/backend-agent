from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories.upstash_redis import UpstashRedisChatMessageHistory
from config import UPSTASH_URL, UPSTASH_TOKEN

def get_memory(session_id:str):
    history = UpstashRedisChatMessageHistory(
        session_id=session_id,
        url=UPSTASH_URL,
        token=UPSTASH_TOKEN
        )  
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages= True,
        chat_memory=history
        )
    
    return memory