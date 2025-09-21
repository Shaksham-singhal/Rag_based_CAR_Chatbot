# --- SQLite Fix for ChromaDB ---
# Ensures Chroma uses a newer sqlite (via pysqlite3-binary) instead of system sqlite 3.34.1
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    print("⚠️ Please run: pip install pysqlite3-binary")


# --- your existing llm_rag.py code continues here ---

import chromadb
from chromadb.utils import embedding_functions
# from google import genai
import google.generativeai as genai
import json
import os
from dotenv import load_dotenv
import re

# Load environment variables from .env file
load_dotenv()

def initialize_clients():
    """Initialize ChromaDB and Google Gemini API clients"""
    # ChromaDB setup
    chroma_db_path = "car_chroma_db" 
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    chroma_client = chromadb.PersistentClient(path=chroma_db_path)

    try:
        collection = chroma_client.get_collection(
            name="car_data_chunks",
            embedding_function=sentence_transformer_ef
        )
    except Exception as e:
        print(f"Error accessing collection: {e}")
        collection = None

    # Google Gemini API setup
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")

    # Initialize Gemini client
    genai.configure(api_key=gemini_api_key)

    return collection, genai

def clean_price(price_str):
    """Convert price string like '₹ 32.8 Lakh' to a structured format"""
    # Extract the numeric part and the denomination (Lakh/Crore)
    if not price_str or not isinstance(price_str, str):
        return price_str

    match = re.search(r'₹\s*([\d.]+)\s*(Lakh|Crore)?', price_str)
    if not match:
        return price_str

    amount = float(match.group(1))
    denomination = match.group(2) if match.group(2) else ""

    if denomination.lower() == "lakh":
        return f"₹{amount} Lakh (₹{amount*100000:,.2f})"
    elif denomination.lower() == "crore":
        return f"₹{amount} Crore (₹{amount*10000000:,.2f})"
    else:
        return f"₹{amount:,.2f}"

def retrieve_context(collection, query, n_results=5, filters=None):
    """Retrieve relevant context from ChromaDB"""
    # No filters case
    if not filters:
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
    else:
        # With filters case
        try:
            results = collection.query(
                query_texts=[query],
                n_results=n_results,
                where=filters
            )
        except ValueError as e:
            print(f"Filter error: {e}. Falling back to query without filters.")
            results = collection.query(
                query_texts=[query],
                n_results=n_results
            )

    # Process results
    contexts = []
    if results and results['documents'] and results['documents'][0]:
        for i in range(len(results['documents'][0])):
            context = {
                "content": results['documents'][0][i],
                "car_name": results['metadatas'][0][i]['car_name'],
                "price": results['metadatas'][0][i]['price'],
                "city": results['metadatas'][0][i]['city'],
                "fuel_type": results['metadatas'][0][i]['fuel_type'],
                "manufacturing_year": results['metadatas'][0][i]['manufacturing_year'],
                "url": results['metadatas'][0][i]['url'],
                "similarity": results['distances'][0][i] if 'distances' in results else None
            }
            contexts.append(context)

    return contexts

def format_context_for_llm(contexts):
    """Format retrieved contexts into a single string for the LLM prompt"""
    if not contexts:
        return "No relevant car information found."

    formatted_context = "RELEVANT CAR LISTINGS:\n\n"

    for i, ctx in enumerate(contexts):
        # Format price for better readability
        clean_price_str = clean_price(ctx.get('price', 'Price not available'))

        formatted_context += f"[CAR {i+1}] {ctx['car_name']}\n"
        formatted_context += f"Price: {clean_price_str}\n"
        formatted_context += f"Location: {ctx.get('city', 'Not specified')}\n"
        formatted_context += f"Fuel Type: {ctx.get('fuel_type', 'Not specified')}\n"
        formatted_context += f"Year: {ctx.get('manufacturing_year', 'Not specified')}\n"
        formatted_context += f"Listing URL: {ctx.get('url', 'Not available')}\n"
        formatted_context += f"Details: {ctx['content']}\n\n"

    return formatted_context

def generate_answer_with_llm(client, query, context):
    """Generate an answer using Google Gemini with the retrieved context"""
    prompt = f"""You are a knowledgeable automotive expert assistant. You help users find and understand information about used cars based on a database of car listings. You'll be given information about various car listings and a user question.

CONTEXT:
{context}

USER QUESTION: {query}

Please answer the question based ONLY on the information provided in the CONTEXT. If the context doesn't contain enough information to fully answer the question, acknowledge this limitation. If the question is about a car not mentioned in the context, state that you don't have information about that specific car.

In your answer:
1. Provide specific details about the cars that match the user's query
2. Compare options if multiple relevant cars are available
3. Highlight important features like price, condition, fuel type, and mileage
4. If appropriate, mention the contact URL for more information
5. Format your response to be easily readable with sections and bullet points when helpful


ANSWER:"""

    try:
        # Initialize Gemini model
        model = client.GenerativeModel('gemini-2.0-flash')
        
        # Generate content
        response = model.generate_content(prompt)
        
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"

def parse_query_for_filters(query):
    """Extract potential filters from a query to narrow down search"""
    filters = []

    brands = ["Toyota", "Honda", "Maruti", "Suzuki", "Hyundai", "Mahindra", "Tata", "Kia",
              "Mercedes", "BMW", "Audi", "Volkswagen", "Ford", "Renault", "Nissan", "MG"]

    for brand in brands:
        if brand.lower() in query.lower():
            filters.append({"car_name": {"$contains": brand}})
            break

    fuel_types = ["Petrol", "Diesel", "CNG", "Electric", "Hybrid"]
    for fuel in fuel_types:
        if fuel.lower() in query.lower():
            filters.append({"fuel_type": fuel})
            break

    cities = ["Delhi", "Mumbai", "Bangalore", "Hyderabad", "Chennai", "Kolkata", "Pune"]
    for city in cities:
        if city.lower() in query.lower():
            filters.append({"city": city})
            break

    year_match = re.search(r'\b(20\d{2})\b', query)
    if year_match:
        filters.append({"manufacturing_year": year_match.group(1)})

    return {"$and": filters} if len(filters) > 1 else filters[0] if filters else None

def car_rag_pipeline(query, explicit_filters=None):
    """Full RAG pipeline combining retrieval and generation"""
    # Initialize clients
    try:
        collection, gemini_client = initialize_clients()
    except Exception as e:
        return f"Error initializing clients: {str(e)}"

    # Parse query for implicit filters
    implicit_filters = parse_query_for_filters(query)

    # Use either explicit or implicit filters, with explicit taking precedence
    filters = None
    if explicit_filters:
        filters = explicit_filters  # Use explicit if provided
    elif implicit_filters:
        filters = implicit_filters  # Otherwise use implicit

    # Step 1: Retrieve relevant contexts
    print(f"Retrieving context for query: '{query}'")
    if filters:
        print(f"Using filters: {filters}")
    contexts = retrieve_context(collection, query, n_results=5, filters=filters)

    if not contexts:
        return "I couldn't find relevant information about this in my car database. Try a different query or check back later as our database is regularly updated."

    # Step 2: Format contexts for the LLM
    formatted_context = format_context_for_llm(contexts)

    # Step 3: Generate answer using Google Gemini
    print("Generating answer with Google Gemini...")
    answer = generate_answer_with_llm(gemini_client, query, formatted_context)

    return answer

# Example usage
if __name__ == "__main__":
    

    # Interactive mode
    print("\n=== Interactive Mode ===")
    print("Enter your car-related questions. Type 'exit' to quit.")

    while True:
        user_query = input("\nWhat would you like to know about cars for sale? ")
        if user_query.lower() in ['exit', 'quit', 'bye']:
            print("Thank you for using the Car RAG system. Goodbye!")
            break

        answer = car_rag_pipeline(user_query)
        print("\n=== Answer ===")
        print(answer)

