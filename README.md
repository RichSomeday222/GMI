# Resume Q&A System with RAG, HyDE and Multi-Turn Chat

A semantic question-answering system built on a personal resume, using RAG, Hypothetical Document Embeddings (HyDE), FAISS dense indexing, BM25 sparse retrieval, and GMI Cloud LLM API.

## Features

- Hybrid retrieval strategy combining:
  - Dense semantic search (FAISS + MiniLM)
  - Sparse keyword search (BM25)
- Hypothetical document expansion (HyDE) to boost recall
- Structured prompt construction including:
  - Retrieved resume snippets
  - Optional HyDE document
  - Multi-turn chat history
- Asynchronous LLM calls via GMI Cloud (DeepSeek-R1)
- Command-line interface supporting multi-turn conversations
- Benchmark script for comparing sequential vs. concurrent queries

## Project Structure

```
rag_demo/
├── data/                   
│   └── resume.pdf           # Input resume file (PDF, TXT, or MD)
├── embeddings/             
│   ├── resume_embeddings.npy
│   └── texts.json           # Mapping from vectors back to text
├── index/                  
│   └── resume.index         # FAISS index file
├── src/
│   ├── preprocess.py        # Load and semantically split resume
│   ├── embed.py             # Generate embeddings from text chunks
│   ├── build_index.py       # Build and persist FAISS index
│   ├── query_engine.py      # Core RAG + HyDE logic and GMI API integration
│   ├── benchmark.py         # Compare sequential vs. asynchronous query performance
│   └── evaluation.py        # Retrieval and generation evaluation scripts
└── .env                     # Your GMI Cloud API key (GMI_API_KEY=…)
```


## Installation

1. Clone the repository and create a virtual environment
   ```bash
   git clone https://github.com/yourusername/rag_demo.git
   cd rag_demo
   python3 -m venv venv
   source venv/bin/activate
   

2. Install dependencies
   ```bash
   pip install -r requirements.txt
3. Create a .env file in the project root:
   ```bash
   GMI_API_KEY=your_api_key_here

## Usage
1. Preprocess the resume
```bash
python src/preprocess.py
2. Generate embeddings
```bash
python src/embed.py
3. Build the FAISS index
```bash
python src/build_index.py
4. Run the multi-turn CLI
```bash
python src/query_engine.py
5. Run the benchmark
```bash
python src/benchmark.py
