from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("API_KEY")
SOURCE_FILE = "source.txt"
INDEX_DIR = "faiss_index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MEMORY_WINDOW_K = 3
MEMORY_WINDOW_SIZE = MEMORY_WINDOW_K * 2
