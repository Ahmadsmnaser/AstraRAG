import logging

from crewai.tools import tool
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core import Settings
import chromadb

from src.agents_src.config.agent_settings import AgentSettings

# Get logger for this module
logger = logging.getLogger(__name__)

# Download & load embedding model
logger.info("Loading HuggingFace embedding model...")
embed_model = HuggingFaceEmbedding()

@tool
def rag_query_tool(query: str) -> dict:
    """
    Answers a query by retrieving relevant documents and generating a response.
    Returns both the generated answer and the source file names from which the information was retrieved.

    Args:
        query (str): The input query string to be processed
        
    Returns:
        dict: A dictionary with the following keys:
            - 'answer' (str): The generated answer to the query
            - 'source_files' (list): A list of source file names from which the information was retrieved
        
    
    Notes:
        - Requires properly configured AgentSettings and access to the vector store.
        - The function loads the embedding model each time it is called.
    """
    settings = AgentSettings()
    vector_store_path = settings.VECTOR_STORE_DIR
    collection_name = settings.COLLECTION_NAME
    settings.llm = Groq(
        model=settings.MODEL_NAME,
        api_key=settings.GROQ_API_KEY,
        temperature=settings.MODEL_TEMPERATURE
    )

    db = chromadb.PersistentClient(path=vector_store_path)
    chroma_collection = db.get_or_create_collection(name=collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
        embed_model=embed_model
        )   

    query_engine = index.as_query_engine(similarity_top_k=3)

    response = query_engine.query(query)

    source_files = {m.get("file_name")for m in getattr(response, "metadata", {}).values()}
    return {
        "answer": response.response,
        "source_files": list(source_files)
    }   
    
    
    