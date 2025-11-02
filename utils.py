import os
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

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
            f.write(
                "=== Example trusted medical knowledge base ===\n"
                "Cough and fever are common in viral infections such as flu or COVID-19.\n"
                "Headache and fatigue can be due to dehydration or tension."
            )

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
