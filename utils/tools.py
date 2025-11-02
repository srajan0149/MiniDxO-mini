from langchain_core.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun

def create_tools(retriever):
    def search_trusted_medical_knowledge(query: str) -> str:
        try:
            docs = retriever.invoke(query)
            if not docs:
                return "No relevant information found in the trusted knowledge base."
            return "\n---\n".join([doc.page_content for doc in docs])
        except Exception as e:
            return f"Error during semantic search: {e}"

    duckduckgo_tool = DuckDuckGoSearchRun()

    tools = [
        Tool(
            name="search_trusted_medical_knowledge",
            func=search_trusted_medical_knowledge,
            description="ALWAYS use this tool FIRST. It semantically searches the trusted, internal medical knowledge base (source.txt) for symptoms and conditions. Use this before any web search."
        ),
        Tool(
            name="duckduckgo_search",
            func=duckduckgo_tool.run,
            description="Use this tool ONLY if 'search_trusted_medical_knowledge' does not provide a useful answer. Use it to find information from credible medical sources (like Mayo Clinic, NIH, MedlinePlus)."
        ),
    ]

    return tools
