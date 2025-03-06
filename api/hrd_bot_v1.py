import os
import requests
import tempfile
from bs4 import BeautifulSoup
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import BSHTMLLoader
from langchain.memory import ConversationBufferMemory
from api.config import GEMINI_API_KEY

# Set website URL
WEBSITE_URL = "https://newhorizoncollegeofengineering.in/"

# Configuration
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
MODEL_NAME = "gemini-2.0-flash-lite"
TEMPERATURE = 0.4

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    temperature=TEMPERATURE,
    google_api_key=GEMINI_API_KEY
)

# Memory for conversation history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Prompt template
PROMPT = PromptTemplate(
    template="""
    Context: {context}

    Question: {question}

    If the question is a greeting like "hello", "hi", "bye", respond in a friendly way.
    
    If the question is about general knowledge (e.g., "what is an electric vehicle?"), provide an answer even if it's not in the context.
    
    If the question requires information from the context and it's available, answer it using only the provided context.

    If the context doesn't contain relevant information, respond naturally, but mention that you don't have enough details.
    """,
    input_variables=["context", "question"]
)

def fetch_html(url):
    """Fetch raw HTML from a given URL."""
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Error fetching the website: {e}")
        return None

def process_website(url):
    """Processes a website and returns text chunks for embedding."""
    html_content = fetch_html(url)
    if not html_content:
        raise ValueError("No content could be fetched from the website.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as temp_file:
        temp_file_path = temp_file.name
        temp_file.write(html_content.encode("utf-8"))

    try:
        loader = BSHTMLLoader(temp_file_path)
        documents = loader.load()
    finally:
        os.unlink(temp_file_path)

    text_splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return text_splitter.split_documents(documents)

def rag_pipeline(query, qa_chain):
    """Retrieve relevant documents and generate a response."""
    response = qa_chain.invoke({"query": query})
    return response['result']

# Initialize embeddings and vector store
try:
    texts = process_website(WEBSITE_URL)
    if not texts:
        raise ValueError("No content found.")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
    vectorstore = FAISS.from_documents(texts, embeddings)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        memory=memory,
        chain_type_kwargs={"prompt": PROMPT}
    )
    print("Chatbot initialized successfully.")
except Exception as e:
    print(f"Error initializing chatbot: {e}")
