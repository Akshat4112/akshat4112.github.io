---
title: "RAG: Making LLMs Smarter With Your Own Data"
date: 2024-07-15T09:00:00+01:00
draft: false
tags: ["RAG", "LLM", "vector-databases", "embeddings", "AI", "information-retrieval", "generative-ai"]
weight: 111
math: true
---

If you've ever asked ChatGPT a question about your company's data or private documents and received a polite "I don't have information about that," you've encountered one of the fundamental limitations of LLMs: **they only know what they were trained on**.

Enter **Retrieval-Augmented Generation (RAG)** — a technique that bridges this gap by connecting LLMs to your own knowledge bases, documents, and data sources.

---

## 🔍 The LLM Knowledge Problem

Large language models are trained on vast datasets from the internet, books, and other public sources. This gives them broad general knowledge but creates three critical limitations:

1. **Knowledge cutoffs**: They don't know about events after their training data ends.
2. **Private information**: They can't access your organization's internal documents.
3. **Hallucinations**: Without access to verified information, they may generate plausible-sounding but incorrect responses.

RAG addresses all three of these challenges with a deceptively simple approach.

---

## 💡 What Is RAG?

**Retrieval-Augmented Generation** combines the power of LLMs with an information retrieval system:

1. **Retrieval**: Find relevant information from your own data sources
2. **Augmentation**: Insert this information into the context window of the LLM
3. **Generation**: Let the LLM create a response based on both its training and the retrieved information

![RAG Architecture Conceptual Flow](/posts/rag_concept.png)

This approach effectively gives the LLM access to knowledge it wasn't originally trained on, without requiring you to fine-tune the model.

---

## ⚙️ How RAG Works (A Technical Overview)

Let's break down the RAG pipeline:

### 1. Document Ingestion and Chunking

First, we prepare our knowledge base:

- **Ingest documents** from various sources (PDFs, web pages, databases, etc.)
- **Extract text** from these documents
- **Chunk the text** into manageable segments (typically 500-1000 tokens)

### 2. Creating Vector Embeddings

Next, we create a searchable representation of our content:

- **Generate vector embeddings** for each chunk using embedding models (e.g., OpenAI's `text-embedding-ada-002` or open-source alternatives like BERT)
- **Store these vectors** in a vector database (like Pinecone, Weaviate, Chroma, or Milvus)

### 3. Retrieval Process

When a user query comes in:

- **Convert the query** to an embedding using the same embedding model
- **Search the vector database** for chunks with similar embeddings (semantic search)
- **Retrieve the most relevant chunks** based on vector similarity

### 4. Context Augmentation and Response Generation

Finally, we put it all together:

- **Construct a prompt** that includes the user question and the retrieved relevant chunks
- **Send this augmented prompt** to the LLM
- **Generate a response** that draws from both the model's training and the retrieved information

---

## 🛠️ Building Blocks of RAG Systems

A robust RAG system requires several components working together:

| Component | Purpose | Popular Options |
|-----------|---------|-----------------|
| **Document Loaders** | Extract text from various file formats | LangChain loaders, Unstructured.io, Apache Tika |
| **Text Chunkers** | Split documents into manageable pieces | LangChain text splitters, Llama Index chunkers |
| **Embedding Models** | Create vector representations of text | OpenAI Embeddings, BERT, Sentence Transformers |
| **Vector Databases** | Store and search vector embeddings | Pinecone, Weaviate, Chroma, Qdrant, Milvus |
| **LLM** | Generate responses using retrieved context | GPT-4, Claude, Mistral, Llama, etc. |
| **Orchestration Layer** | Coordinate the entire pipeline | LangChain, LlamaIndex, custom code |

---

## 🚀 Advanced RAG Techniques

The basic RAG approach works well, but several advanced techniques can significantly improve performance:

### 1. Hybrid Search

Combine vector similarity with traditional keyword search (BM25) for better retrieval precision.

```python
# Pseudo-code for hybrid search
def hybrid_search(query, alpha=0.5):
    vector_results = vector_search(query)  # semantic search
    keyword_results = bm25_search(query)   # keyword search
    
    # Combine results with weighting factor alpha
    combined_results = []
    for doc in set(vector_results + keyword_results):
        score = alpha * doc.vector_score + (1-alpha) * doc.keyword_score
        combined_results.append((doc, score))
    
    return sorted(combined_results, key=lambda x: x[1], reverse=True)
```

### 2. Query Transformations

Expand or rewrite the user's query to improve retrieval effectiveness.

- **Query Expansion**: Generate multiple variations of the original query
- **HyDE** (Hypothetical Document Embeddings): Generate a hypothetical answer first, then use that to search
- **Sub-query Decomposition**: Break complex queries into simpler sub-queries

### 3. Re-ranking

Apply a second pass of ranking to the initial retrieval results.

```python
# Pseudo-code for re-ranking
def rerank(query, initial_results, reranker_model):
    scored_pairs = []
    for doc in initial_results:
        # Score query-document relevance more precisely
        relevance_score = reranker_model.score(query, doc.content)
        scored_pairs.append((doc, relevance_score))
    
    return sorted(scored_pairs, key=lambda x: x[1], reverse=True)[:top_k]
```

### 4. Contextual Compression

Trim retrieved documents to only the most relevant parts before sending to the LLM.

```python
def compress_context(query, retrieved_chunks, compressor_model):
    compressed_chunks = []
    for chunk in retrieved_chunks:
        # Extract only the most relevant sentences
        relevant_content = compressor_model.extract_relevant(query, chunk)
        compressed_chunks.append(relevant_content)
    
    return compressed_chunks
```

### 5. Multi-step RAG

Chain multiple retrievals and generation steps for complex reasoning.

---

## 📈 Evaluating RAG Systems

Measuring RAG performance involves several metrics:

1. **Retrieval Metrics**: How well does the system retrieve relevant information?
   - Precision@k, Recall@k, Mean Reciprocal Rank (MRR)
   - NDCG (Normalized Discounted Cumulative Gain)

2. **Generation Metrics**: How accurate and useful are the final responses?
   - Faithfulness (does the answer stick to facts in the retrieved context?)
   - Answer relevance (does it actually answer the question?)
   - Hallucination rate (does it make up information?)

3. **End-to-end Metrics**: How does the system perform as a whole?
   - User satisfaction ratings
   - Task completion rate
   - Response latency

---

## 🪤 Common RAG Pitfalls and Solutions

As I've implemented RAG systems, I've encountered several common challenges:

| Challenge | Description | Potential Solution |
|-----------|-------------|-------------------|
| **Chunking Issues** | Chunks too large or small, or cutting across logical boundaries | Use semantic chunking based on document structure |
| **Context Limit** | Retrieved context exceeds LLM token limits | Implement better ranking and contextual compression |
| **Retrieval Mismatch** | Vector search returns semantically similar but factually irrelevant results | Use hybrid search, re-ranking, or filter with metadata |
| **Prompt Engineering** | Poor instructions to the LLM about how to use the context | Develop explicit prompts that guide the model to cite sources |
| **Answer Synthesis** | LLM struggles to synthesize information from multiple chunks | Implement chain-of-thought prompting or multi-step reasoning |

---

## 🏗️ Building a Basic RAG System

Here's a simplified example of building a RAG system using Python with LangChain:

```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader

# 1. Load documents
loader = DirectoryLoader('./documents/', glob="**/*.pdf")
documents = loader.load()

# 2. Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

# 3. Create embeddings and vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(chunks, embeddings)

# 4. Create a retrieval chain
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0),
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5})
)

# 5. Ask questions
query = "What are the key findings from our Q2 financial report?"
response = qa_chain.run(query)
print(response)
```

---

## 🔮 The Future of RAG

RAG continues to evolve rapidly:

- **Multi-modal RAG**: Retrieving and reasoning over images, audio, and video alongside text
- **Agent-based RAG**: Autonomous systems that determine when and what to retrieve
- **LLM-powered retrieval**: Using LLMs themselves to improve the retrieval process
- **Fine-tuned retrievers**: Custom models specifically optimized for domain-specific retrieval

---

## 🧠 Final Thoughts

RAG represents a paradigm shift in how we deploy LLMs in real-world applications. Rather than trying to train models on all possible information, we're building systems that can access, retrieve, and reason over external knowledge on demand.

This approach offers several advantages:

- **Lower cost**: No need to continuously retrain large models
- **Better accuracy**: Grounded responses based on verified information
- **Freshness**: Easy to update knowledge without touching the model
- **Privacy**: Keep sensitive data separate from the model itself

For organizations looking to leverage LLMs with their own data, RAG provides the most practical path forward—connecting the general capabilities of foundation models with the specific knowledge needed for your unique use cases.

— **Akshat** 