import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_community.tools import TavilySearchResults

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool

from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories.upstash_redis import UpstashRedisChatMessageHistory


from dotenv import load_dotenv
load_dotenv()

history = UpstashRedisChatMessageHistory(
    session_id="test",
    url=os.getenv("UPSTASH_REDIS_URL"),
    token=os.getenv("UPSTASH_REDIS_TOKEN")
)


pinecone =Pinecone()
index = pinecone.Index("langchain")

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
)
vectorStore = PineconeVectorStore(
    index,
    embeddings
)

retrieve = vectorStore.as_retriever(search_kwargs={"k": 2})

model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
)

prompt = ChatPromptTemplate.from_messages(
    [
    ("system", "You are a friendly assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages= True,
    chat_memory=history
)

search = TavilySearchResults()
retriever_tool = create_retriever_tool(
    retriever=retrieve,
    name="retriever",
    description="Use this tool to retrieve information from the database "
)
tools= [search, retriever_tool]

agent = create_openai_functions_agent(
    llm=model,
    prompt=prompt,
    tools=tools,
)

agentExecutor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory
    # verbose=True
)

def process_chat(agentExecutor, user_input):
    response = agentExecutor.invoke({
        "input": user_input,
    })
    return(response['output'])

if __name__ == "__main__":

    while True:
        user_input = input("You: ")
        response = process_chat(agentExecutor, user_input)
        print("Max: ", response)
       