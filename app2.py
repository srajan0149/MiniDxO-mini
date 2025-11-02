import os
import streamlit as st
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import Tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter

from config import API_KEY

SOURCE_FILE = "source.txt"
INDEX_DIR = "faiss_index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MEMORY_WINDOW_K = 3
MEMORY_WINDOW_SIZE = MEMORY_WINDOW_K * 2

def get_or_create_vector_store(source_path, index_path, embedding_model):
    if os.path.exists(index_path):
        try:
            vector_store = FAISS.load_local(
                index_path, 
                embedding_model,
                allow_dangerous_deserialization=True
            )
            return vector_store
        except Exception:
            pass
    if not os.path.exists(source_path):
        with open(source_path, "w") as f:
            f.write("=== Example trusted medical knowledge base ===\nCough and fever are common in viral infections such as flu or COVID-19.\nHeadache and fatigue can be due to dehydration or tension.")
    loader = TextLoader(source_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(
        separator="===",
        chunk_size=500,
        chunk_overlap=50
    )
    docs = text_splitter.split_documents(documents)
    vector_store = FAISS.from_documents(docs, embedding_model)
    vector_store.save_local(index_path)
    return vector_store

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vector_store = get_or_create_vector_store(SOURCE_FILE, INDEX_DIR, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

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

AGENT_SYSTEM_PROMPT = """
You are MiniDxO, a friendly and transparent AI diagnostic assistant.
Your goal is to simulate a doctor's reasoning process step-by-step.
Your memory is limited to only the last 3 interactions, so be concise.

Your process MUST be:

1.  *GREET & QUESTION:* The user will state their main symptom. Greet them and ask 1-2 clarifying questions to get more details (e.g., "How long have you had this?", "Do you have a fever?").
    * *Constraint:* Do not ask more than 3-4 questions in total for the entire interaction.

2.  *CHECK INTERNAL KNOWLEDGE:* After you have 2-3 key symptoms (e.g., "cough" and "fever"), your FIRST action MUST be to use the search_trusted_medical_knowledge tool. This is your trusted source.
    * The query you provide to this tool should be a summary of the user's symptoms.
    * Analyze the tool's output. Does it match the user's symptoms?

3.  *CHECK EXTERNAL KNOWLEDGE (If needed):* ONLY if search_trusted_medical_knowledge returns "no relevant information" or is insufficient, your SECOND action should be to use duckduckgo_search.
    * Search for the cluster of symptoms (e.g., "sudden fever and body aches").
    * Prioritize information from credible sources like "Mayo Clinic", "NIH", or "MedlinePlus" in your search query.

4.  *EXPLAIN & DIAGNOSE:* Once you have gathered information (from questions + one or both tools), you MUST explain your thought process clearly before giving a probable diagnosis.
    * Start with "Here is my thought process:"
    * List the key symptoms the user provided.
    * State what your search tools found (e.g., "My internal knowledge from semantic search suggests...").
    * Conclude with a probable, non-definitive diagnosis (e.g., "Based on this, it's possible you are experiencing...").

You must follow this order. Be empathetic and clear.
"""

llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    api_key=API_KEY,
    temperature=0.0,
    max_tokens=1024,
)

agent = create_agent(
    llm,
    tools=tools,
    system_prompt=AGENT_SYSTEM_PROMPT
)

st.set_page_config(page_title="MiniDxO - AI Diagnostic Assistant", page_icon="🩺", layout="centered")

st.markdown("""
    <style>
    .chat-bubble-user {
        background-color: #DCF8C6;
        padding: 10px 15px;
        border-radius: 10px;
        margin: 5px 0;
        text-align: right;
    }
    .chat-bubble-ai {
        background-color: #F1F0F0;
        padding: 10px 15px;
        border-radius: 10px;
        margin: 5px 0;
        text-align: left;
    }
    .stTextInput>div>div>input {
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🩺 MiniDxO — AI Diagnostic Assistant")
st.markdown("### Talk to MiniDxO about your symptoms 💬")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation" not in st.session_state:
    st.session_state.conversation = []

for msg in st.session_state.conversation:
    if msg["role"] == "user":
        st.markdown(f"<div class='chat-bubble-user'>👤 {msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-bubble-ai'>🤖 {msg['content']}</div>", unsafe_allow_html=True)

user_input = st.chat_input("Describe your symptoms...")

if user_input:
    st.session_state.conversation.append({"role": "user", "content": user_input})
    with st.spinner("MiniDxO is thinking..."):
        try:
            messages_for_invoke = [HumanMessage(content=m["content"]) for m in st.session_state.conversation[-MEMORY_WINDOW_SIZE:]]
            response = agent.invoke({"messages": messages_for_invoke})
            all_messages = response["messages"]
            ai_message = all_messages[-1]
            ai_response = ai_message.content

            print("\n=== Multi-Doctor Reasoning Session ===")
            dr_hypothesis = llm.invoke([HumanMessage(content=f"User symptoms: {user_input}. As Dr. Hypothesis, propose top 3 possible causes.")])
            hypothesis = dr_hypothesis.content
            print(f"\nDr. Hypothesis:\n{hypothesis}")

            for i in range(5):
                dr_challenge = llm.invoke([HumanMessage(content=f"As Dr. Challenge, logically challenge Dr. Hypothesis's possibilities:\n{hypothesis}")])
                challenge = dr_challenge.content
                print(f"\nDr. Challenge (iteration {i+1}):\n{challenge}")

                dr_checklist = llm.invoke([HumanMessage(content=f"As Dr. Checklist, ensure clinical consistency between these points and refine final thought:\nHypothesis: {hypothesis}\nChallenge: {challenge}")])
                checklist = dr_checklist.content
                print(f"\nDr. Checklist (iteration {i+1}):\n{checklist}")

                if "final" in checklist.lower() or "conclusion" in checklist.lower() or i == 4:
                    final_round = checklist
                    break
                hypothesis = checklist

            print("\n=== Final Consensus ===")
            print(final_round)
            ai_response += f"\n\n🤝 **Final Doctor Consensus:**\n{final_round}"

        except Exception as e:
            ai_response = f"⚠ Error: {e}"
    st.session_state.conversation.append({"role": "ai", "content": ai_response})
    st.rerun()

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3774/3774294.png", width=100)
    st.markdown("### About MiniDxO")
    st.write("""
    MiniDxO is a friendly AI health assistant that simulates a doctor’s reasoning.
    It first checks its *trusted internal knowledge*, then searches the web (if needed).
    Always consult a real doctor for medical advice.
    """)
    st.markdown("---")
    st.caption("Health Guider")