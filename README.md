<img width="1917" height="912" alt="image" src="https://github.com/user-attachments/assets/d68b0952-6b45-4cf3-a81b-1545be0c5840" />


# CAR CHATBOT

## Overview
This project is a chatbot designed for evaluating car-related queries. It leverages a combination of retrieval-augmented generation (RAG) and NLP-based evaluation metrics.

## Project Structure
```
CAR CHATBOT/
├── Car Chatbot/
│   ├── car_chroma_db/
│   ├── car_env/
│   ├── data/
│   │   ├── evaluation_results_data_in_batches/
│   │   ├── cartrade_cars_chunked.json
│   │   ├── cartrade_cars_final.json
│   │   ├── cartrade_cars.json
│   │   └── evaluation_results_summary.json
│   ├── src/
│   │   ├── _pycache_/
│   │   ├── Car Chatbot/
│   │   ├── car_chroma_db/
│   │   ├── .env
│   │   ├── chunking.py
│   │   ├── embedding_store.py
│   │   ├── eval_testing_03.py
│   │   ├── llm_rag.py
│   │   ├── scrapper.py
│   │   └── app.py
│   └── test/
│       ├── _pycache_/
│       ├── _init_.py
│       ├── test_eval_testing_03.py
│       ├── test_llm_rag.py
│       ├── test_streamlit.py
│       ├── evaluation_results_1.json
│       ├── requirement.txt
│       
```

## Setup Instructions

1. **Install Dependencies:**
   ```bash
   pip install -r requirement.txt
   ```

2. **Run the Chatbot:**
   ```bash
   python llm_rag.py
   ```

3. **Run the UI Interface:**
   ```bash
   streamlit run app.py
   ```

4. **Run Evaluation:**
   ```bash
   python eval_testing_03.py
   ```
   This script calculates BERT, ROUGE, METEOR, and F1 scores for evaluation.

   

## Data Sources
- **cartrade_cars_chunked.json** contains the actual data fed to the model.
- The dataset has been chunked and stored for efficient retrieval.

## Description of Key Files
- **llm_rag.py** - Contains the core logic for the chatbot.
- **scrapper.py** - Scrapes data from car-related sources.
- **streamlit.py** - Streamlit-based UI for interacting with the chatbot.
- **eval_testing_03.py** - Computes NLP-based evaluation metrics.



