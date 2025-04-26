from config import LLM_MODEL
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_functions_agent,AgentExecutor
from tools import get_tools
from memory import get_memory

def agent(session_id:str):

    tools = get_tools()

    model = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0.7,
        )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a friendly assistant."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
    
    memory = get_memory(session_id)

    agent = create_openai_functions_agent(
        llm=model,
        prompt=prompt,
        tools=tools,
        )   

    agentExecutor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory
        )

    return agentExecutor

def process_chat(user_input, session_id='123'):
    
    response = agent(session_id).invoke({
        "input": user_input,
    })
    return(response['output'])

if __name__ == "__main__":

    while True:
        user_input = input("You: ")
        response = process_chat(user_input)
        print("Max: ", response)
       