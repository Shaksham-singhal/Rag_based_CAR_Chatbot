import json
import os
import logging
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Set up paths - modify these to match your environment
chunked_data_path = "..\data\cartrade_cars_chunked.json"  # Path to your chunked car data
chroma_db_path = "car_chroma_db"  # Where you want to store the ChromaDB

# Create directory for Chroma DB if it doesn't exist
os.makedirs(chroma_db_path, exist_ok=True)

# Initialize ChromaDB with sentence transformers
logger.info("Initializing ChromaDB with sentence-transformers...")
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"  # This is a good general-purpose embedding model
)

# Create Chroma client and collection
client = chromadb.PersistentClient(path=chroma_db_path)
collection = client.get_or_create_collection(
    name="car_data_chunks",
    embedding_function=sentence_transformer_ef,
    metadata={"description": "Used car data chunks for RAG system"}
)

# Load chunked data
logger.info(f"Loading chunked data from {chunked_data_path}...")
with open(chunked_data_path, 'r', encoding='utf-8') as f:
    chunks = json.load(f)

logger.info(f"Found {len(chunks)} chunks")

# Function to clean metadata (replace None values)
def clean_metadata(metadata):
    """Replace None values with appropriate defaults for ChromaDB compatibility."""
    return {
        key: (str(value) if value is not None else "Unknown")  # Convert None to string
        for key, value in metadata.items()
    }

# Process chunks in batches for ChromaDB
batch_size = 100
total_batches = (len(chunks) + batch_size - 1) // batch_size  # Ceiling division

logger.info(f"Processing chunks in {total_batches} batches...")

for batch_index in tqdm(range(total_batches), desc="Uploading to ChromaDB"):
    start_idx = batch_index * batch_size
    end_idx = min((batch_index + 1) * batch_size, len(chunks))
    batch_chunks = chunks[start_idx:end_idx]

    ids = []
    documents = []
    metadatas = []

    for chunk in batch_chunks:
        chunk_id = chunk["chunk_id"]
        ids.append(chunk_id)
        documents.append(chunk["text"])

        # Clean metadata before inserting
        raw_metadata = chunk.get("metadata", {})  # Use .get() to avoid KeyError
        metadata = clean_metadata(raw_metadata)
        metadata["chunk_index"] = chunk["chunk_index"]  # Ensure chunk_index is included

        metadatas.append(metadata)

    # Add batch to collection (handle potential duplicates)
    try:
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
    except Exception as e:
        logger.warning(f"Error in batch {batch_index}: {str(e)}")

        # Try adding one by one to skip duplicates
        for i in range(len(ids)):
            try:
                collection.add(
                    ids=[ids[i]],
                    documents=[documents[i]],
                    metadatas=[metadatas[i]]
                )
            except Exception as sub_e:
                if "already exists" in str(sub_e):
                    logger.info(f"Skipping duplicate ID: {ids[i]}")
                else:
                    logger.error(f"Error adding document {ids[i]}: {str(sub_e)}")

logger.info(f"Completed loading {collection.count()} chunks into ChromaDB")

# Test a query
logger.info("\nTesting a sample query...")
query_text = "White Toyota Fortuner diesel car in Delhi"
results = collection.query(
    query_texts=[query_text],
    n_results=3
)

if results and results.get("documents") and results["documents"][0]:
    logger.info("Sample results:")
    for i in range(len(results["documents"][0])):
        logger.info(f"\nResult {i+1}:")
        logger.info(f"Car: {results['metadatas'][0][i]['car_name']}")
        logger.info(f"Price: {results['metadatas'][0][i]['price']}")
        logger.info(f"City: {results['metadatas'][0][i]['city']}")
        logger.info(f"Fuel: {results['metadatas'][0][i]['fuel_type']}")
        logger.info(f"Year: {results['metadatas'][0][i]['manufacturing_year']}")
        logger.info(f"Content snippet: {results['documents'][0][i][:150]}...")
else:
    logger.warning("No results found for sample query")

logger.info("\nChromaDB setup complete! You can now query the database.")

# Function to query the database
def query_car_database(query_text, n_results=5):
    """Query the car database with a natural language query"""
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    return results
