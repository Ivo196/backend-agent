from langchain_community.tools import TavilySearchResults
from langchain.tools.retriever import create_retriever_tool
from retriever import get_retriever

def get_tools():
    search = TavilySearchResults()
    retriever_tool = create_retriever_tool(
        retriever=get_retriever(),
        name="retriever",
        description="Use this tool to retrieve information from the database "
    )
    return [search, retriever_tool]