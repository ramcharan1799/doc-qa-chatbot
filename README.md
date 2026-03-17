# Document Q&A Chatbot

An AI-powered chatbot that answers questions about any PDF document using RAG (Retrieval-Augmented Generation).

## How it works

1. Upload any PDF via the sidebar
2. The app chunks the text and builds a FAISS vector index using sentence-transformers
3. When you ask a question, the most relevant chunks are retrieved and sent to GPT-3.5
4. The answer is grounded in the document and cites the page number

## Tech stack

- **OpenAI GPT-3.5** — answer generation
- **sentence-transformers** (all-MiniLM-L6-v2) — local embeddings, runs free
- **FAISS** — vector similarity search
- **Streamlit** — UI and deployment
- **pypdf** — PDF parsing

## Project structure
```
doc-qa-chatbot/
├── app.py              # Streamlit UI
├── utils/
│   ├── loader.py       # PDF parsing and chunking
│   └── qa_engine.py    # Embeddings, FAISS index, OpenAI Q&A
├── requirements.txt
└── .env                # API key (not committed)
```

## Run locally
```
git clone https://github.com/ramcharan1799/doc-qa-chatbot
cd doc-qa-chatbot
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
echo "OPENAI_API_KEY=your-key-here" > .env
streamlit run app.py
```

## What I learned

- How to parse and chunk PDF documents for LLM consumption
- How to build a vector index with FAISS and sentence-transformers
- How to implement a RAG pipeline connecting retrieval to generation
- How to build and deploy an interactive AI app with Streamlit

## Author

Built as part of my AI Engineer learning journey — roadmap.sh/ai-engineer
