from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_community.tools import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool


from dotenv import load_dotenv
load_dotenv()

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
    tools=tools
)

agentExecutor = AgentExecutor(
    agent=agent,
    tools=tools
    # verbose=True
)

def process_chat(agentExecutor, user_input, chat_history):
    response = agentExecutor.invoke({
        "input": user_input,
        "chat_history": chat_history
    })
    return(response['output'])

if __name__ == "__main__":

    chat_history = [ ]
    while True:
        user_input = input("You: ")
        response = process_chat(agentExecutor, user_input, chat_history)
        print("Max: ", response)
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))