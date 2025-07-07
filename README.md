# RAG-Powered-Chatbot
## Task1 - Initial EDA

In the initial exploration of consumer complaints, we standardized the product labels into five core categories: Credit Card, Personal Loan, Buy Now, Pay Later (BNPL), Savings Account, and Money Transfers. We filtered the dataset to include only complaints related to these products and removed entries with missing or empty narrative text. This provided a cleaner subset focused on high-signal, text-rich complaints.


We found that [insert dominant product here] was the most commonly reported product category. Narrative lengths varied significantly — while the median complaint was around 17 words, some extended beyond 2411 words, indicating a wide range in complaint detail. This suggests the need for text chunking in downstream NLP tasks. The cleaned and filtered data is now ready for embedding and retrieval pipeline construction.

## Task 2 - Chunking, Embedding, and Indexing

### Chunking Strategy

- **Tool Used**: `RecursiveCharacterTextSplitter` from LangChain  
- **Parameters**:  
  - `chunk_size`: 500  
  - `chunk_overlap`: 100  
- **Justification**:  
  - Ensures semantic coherence within chunks  
  - Overlap preserves context across chunks  
  - Avoids mid-sentence splits, enhancing embedding quality  

### Embedding Model

- **Model**: `sentence-transformers/all-MiniLM-L6-v2`  
- **Why This Model**:  
  - Optimized for sentence-level semantic similarity  
  - Lightweight (~80MB) and fast  
  - Strong benchmark performance and industry adoption  
  - Easy integration via `HuggingFaceEmbeddings` in LangChain
