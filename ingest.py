import os
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is not set in the environment variables.")

# This MUST match the model in app.py
EMBEDDINGS_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# This MUST match the name in Pinecone and app.py
INDEX_NAME = "medicalbot"
# This is your local folder with the PDF
DATA_PATH = "data"

def load_pdf_file(data_path):
    loader = DirectoryLoader(
        data_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents

def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)
    return embeddings

if __name__ == "__main__":
    print("Loading documents from 'data' folder...")
    extracted_data = load_pdf_file(DATA_PATH)
    print(f"Loaded {len(extracted_data)} documents.")
    
    print("Splitting text into chunks...")
    text_chunks = text_split(extracted_data)
    print(f"Created {len(text_chunks)} text chunks.")

    print("Downloading embedding model...")
    embeddings = download_hugging_face_embeddings()
    print("Embedding model downloaded successfully.")

    print(f"Checking for Pinecone index '{INDEX_NAME}' and ingesting data...")
    try:
        # This will add documents to your existing index
        vectorstore = PineconeVectorStore.from_documents(
            text_chunks,
            embeddings,
            index_name=INDEX_NAME
        )
        print(f"Successfully ingested {len(text_chunks)} documents into Pinecone.")

    except Exception as e:
        print(f"An error occurred while ingesting data into Pinecone: {e}")
        print("******************************************************************")
        print(">>> COMMON ERROR: Did you create the Pinecone index with the correct name?")
        print(f">>>               Index Name: {INDEX_NAME}")
        print(">>>               Required Dimensions: 384")
        print("******************************************************************")