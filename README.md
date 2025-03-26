# Multi-AI Agent with RAG

## Overview
The **Multi-AI Agent with RAG** is a retrieval-augmented generation (RAG) system designed for efficient knowledge retrieval and structured multi-agent collaboration. It integrates:

- **LangGraph** for managing structured workflows between AI agents.
- **AstraDB** as a vector database for document storage and retrieval.
- **Groq API** to access Llama 3.1 for response generation.
- **LangChain** to facilitate data processing, retrieval, and agent interactions.

## Features
- **Multi-Agent Collaboration**: Uses LangGraph to orchestrate AI agents for intelligent routing and retrieval.
- **Retrieval-Augmented Generation (RAG)**: Enhances knowledge retrieval through AstraDB and vector embeddings.
- **Structured Query Routing**: Routes queries to a vector store or Wikipedia based on context.
- **Graph-based Execution Flow**: Uses LangGraph to manage decision-making and task execution.

## Technologies Used
- **LangChain**
- **LangGraph**
- **AstraDB**
- **Groq API (Llama 3.1)**
- **Hugging Face Embeddings**
- **ChromaDB**
- **Wikipedia & Arxiv APIs**
- **Python**

## Installation
To set up the environment, install the required dependencies:

```bash
pip install langchain langgraph cassio langchain_community tiktoken langchain-groq langchainhub chromadb langchain_huggingface
pip install arxiv wikipedia
```

## Setup
### Connecting AstraDB
```python
import cassio
ASTRA_DB_APPLICATION_TOKEN = "your-astra-token"
ASTRA_DB_ID = "your-astra-db-id"
cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)
```

### Loading Documents
```python
from langchain_community.document_loaders import WebBaseLoader
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/"
]
docs = [WebBaseLoader(url).load() for url in urls]
```

### Vector Embedding & Storage
```python
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.cassandra import Cassandra
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
astra_vector_store = Cassandra(
    embedding=embeddings, table_name="qa_mini_demo", session=None, keyspace=None
)
astra_vector_store.add_documents(docs)
retriever = astra_vector_store.as_retriever()
```

### Query Routing & LLM Integration
```python
from langchain_groq import ChatGroq
import os
os.environ["GROQ_API_KEY"] = "your-groq-api-key"
llm = ChatGroq(groq_api_key=os.environ["GROQ_API_KEY"], model_name="Gemma2-9b-It")
```

### Graph-Based Execution Flow
```python
from langgraph.graph import StateGraph, START, END
workflow = StateGraph(GraphState)
workflow.add_conditional_edges(
    START,
    route_question,
    {
        "wiki_search": "wiki_search",
        "vectorstore": "retrieve",
    },
)
workflow.add_edge("retrieve", END)
workflow.add_edge("wiki_search", END)
app = workflow.compile()
```

## Usage
To run a query:
```python
inputs = {"question": "What is agent memory?"}
for output in app.stream(inputs):
    print(output)
```

## Future Improvements
- Expand agent memory capabilities.
- Enhance LLM fine-tuning for better response accuracy.
- Integrate additional APIs for diverse knowledge retrieval.

## License
This project is licensed under the MIT License.

