import os
import streamlit as st
from langchain_anthropic import ChatAnthropic
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import API_KEY, SOURCE_FILE, INDEX_DIR, EMBEDDING_MODEL, MEMORY_WINDOW_SIZE
from utils.vector_store import get_or_create_vector_store
from utils.tools import create_tools
from utils.prompt import AGENT_SYSTEM_PROMPT

# -------------------------------
# Setup
# -------------------------------
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vector_store = get_or_create_vector_store(SOURCE_FILE, INDEX_DIR, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})
tools = create_tools(retriever)

llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    api_key=API_KEY,
    temperature=0.0,
    max_tokens=1024,
)

agent = create_agent(llm, tools=tools, system_prompt=AGENT_SYSTEM_PROMPT)

# -------------------------------
# Streamlit UI
# -------------------------------
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

# Chat display
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
        except Exception as e:
            ai_response = f" Error: {e}"

    st.session_state.conversation.append({"role": "ai", "content": ai_response})
    st.rerun()

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3774/3774294.png", width=100)
    st.markdown("### About MiniDxO")
    st.write("""
    MiniDxO is a friendly AI health assistant that simulates a doctor’s reasoning.
    It first checks its **trusted internal knowledge**, then searches the web (if needed).
    Always consult a real doctor for medical advice.
    """)
    st.markdown("---")
    st.caption("Health Guider")
