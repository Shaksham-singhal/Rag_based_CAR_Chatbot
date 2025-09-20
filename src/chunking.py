import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
import uuid

# 1. Load the car data from the JSON file
def load_car_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return json.load(file)

# 2. Convert cars to structured text format for better chunking
def format_car_for_chunking(car):
    """Format a car JSON object into a structured text representation for chunking."""
    formatted_text = f"CAR: {car['car_name']}\n"
    formatted_text += f"PRICE: {car['price']}\n"

    # Add all details
    formatted_text += "DETAILS:\n"
    for key, value in car['details'].items():
        formatted_text += f"  {key.replace('_', ' ').title()}: {value}\n"

    # Add seller remarks if available
    if car['seller_remarks']:
        formatted_text += f"SELLER REMARKS: {car['seller_remarks']}\n"

    formatted_text += f"URL: {car['url']}\n"

    return formatted_text

# 3. Create individual JSON objects for each car with its metadata
def create_car_documents(cars):
    """Create separate documents for each car with metadata."""
    documents = []

    for car in cars:
        # Create formatted text for the car
        text = format_car_for_chunking(car)

        # Extract metadata safely using .get() to avoid KeyErrors
        details = car.get("details", {})
        doc = {
            "text": text,
            "metadata": {
                "car_name": car.get("car_name", "Unknown"),
                "price": car.get("price", "Unknown"),
                "city": details.get("city", "Unknown"),
                "fuel_type": details.get("fuel_type", "Unknown"),
                "manufacturing_year": details.get("manufacturing_year", "Unknown"),
                "url": car.get("url", "Unknown")
            }
        }

        documents.append(doc)

    return documents


# 4. Apply LangChain chunking
def chunk_car_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Split documents into chunks using LangChain's RecursiveCharacterTextSplitter."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    chunked_documents = []

    for doc in documents:
        # Split the document text into chunks
        chunks = text_splitter.split_text(doc["text"])

        # Create new documents for each chunk with the original metadata
        for i, chunk_text in enumerate(chunks):
            chunk_doc = {
                "chunk_id": str(uuid.uuid4()),
                "chunk_index": i,
                "text": chunk_text,
                "metadata": doc["metadata"]
            }
            chunked_documents.append(chunk_doc)

    return chunked_documents

# 5. Main function to process and save chunked data
def process_and_save_car_chunks(input_filepath, output_filepath, chunk_size=1000, chunk_overlap=200):
    """Process the car data JSON and save chunked data to a new JSON file."""
    # Load cars
    cars = load_car_data(input_filepath)
    print(f"Loaded {len(cars)} cars from {input_filepath}")

    # Create structured documents
    car_documents = create_car_documents(cars)
    print(f"Created {len(car_documents)} car documents")

    # Apply chunking
    chunked_documents = chunk_car_documents(car_documents, chunk_size, chunk_overlap)
    print(f"Generated {len(chunked_documents)} chunks")

    # Save to JSON file
    with open(output_filepath, 'w', encoding='utf-8') as file:
        json.dump(chunked_documents, file, ensure_ascii=False, indent=2)

    print(f"Saved chunked data to {output_filepath}")

    # Print sample chunk
    if chunked_documents:
        print("\nSample chunk:")
        print(json.dumps(chunked_documents[0], indent=2))

    return chunked_documents

# Example usage
if __name__ == "__main__":
    input_file = "data\cartrade_cars_final.json"
    output_file = "data\cartrade_cars_chunked.json"

    # Process with custom chunk size and overlap
    chunked_data = process_and_save_car_chunks(
        input_file,
        output_file,
        chunk_size=1500,  # Adjust based on your needs
        chunk_overlap=150  # Adjust based on your needs
    )