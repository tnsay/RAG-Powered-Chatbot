#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# get_ipython().system('pip install -q transformers accelerate bitsandbytes langchain chromadb sentence-transformers')
# get_ipython().system('pip install -U langchain langchain-community')
# get_ipython().system('pip install langchain-chroma')


# In[ ]:


# from google.colab import drive
# drive.mount('/content/drive')


# In[ ]:


from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline
from langchain.docstore.document import Document

import os
from dotenv import load_dotenv

env_path = "/content/drive/MyDrive/myenv/.env"
load_dotenv(dotenv_path=env_path)


from huggingface_hub import login
login(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))


from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import pipeline

# === 5. Load Vector Store (From Drive, Created in Task 2) ===
VECTOR_STORE_DIR = "/content/drive/MyDrive/rag/vector_store"

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load the real vector store (persisted)
vectorstore = Chroma(persist_directory=VECTOR_STORE_DIR, embedding_function=embedding_model)

# === 6. Load LLM Generator ===
generator = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.1", device_map="auto")

# === 7. Prompt Template ===
PROMPT_TEMPLATE = """
You are a financial analyst assistant for CrediTrust.
Your task is to answer questions about customer complaints.

Use the following retrieved complaint excerpts to formulate your answer.
If the context doesn't contain the answer, say:
"I don't have enough information from the complaints to answer that."

Context:
{context}

Question: {question}

Answer:
"""

# === 8. Retrieval + Generation Functions ===
def retrieve_chunks(query: str, top_k: int = 5):
    return vectorstore.similarity_search(query, k=top_k)

def generate_answer(question: str, retrieved_docs, prompt_template: str = PROMPT_TEMPLATE):
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    prompt = prompt_template.format(context=context, question=question)
    result = generator(prompt, max_new_tokens=300, do_sample=True, temperature=0.7)
    return result[0]['generated_text'].split("Answer:")[-1].strip()

def run_rag_pipeline(user_question: str):
    print("\n--- RAG Pipeline ---")
    print(f"User Question: {user_question}")

    retrieved = retrieve_chunks(user_question, top_k=5)
    print("\nRetrieved Chunks:")
    for i, doc in enumerate(retrieved):
        print(f"[{i+1}] {doc.page_content[:200]}...")

    answer = generate_answer(user_question, retrieved)
    print("\nGenerated Answer:")
    print(answer)
    return answer, retrieved

# === 9. Run a Test ===
run_rag_pipeline("Why was my account locked after a check deposit?")


# In[ ]:


questions = [
    "i made the mistake of using my wellsfargo debit card to depsit funds into xxxxxxxx atm machine outside their branch.",
    "i have an unfamiliar inquieries report. i contacted the bank",
    "company has responded to the consumer and the cfpb and chooses not to provide a public response",
    "didn t receive advertised or promotional terms",
    "i want my 5.00 back!",
    "i have a credit card with navy federal credit union. i have multiple 20 fraudulent charges from a company called xxxx xxxx. i have been trying to get this resolved with my bank since xxxx but they keep denying my claim. navy federal told me it was a money order company overseas. i have contacted xxxx xxxx but i never can get in touch with a person.",
    "i have been dealing with an on going fraud issue with xxxx bank xxxx which due to their ineptitude"
]

for q in questions:
    print("="*80)
    run_rag_pipeline(q)


# In[ ]:


# !git config --global user.name "tnsay"
# !git config --global user.email "tnsaydagne@gmail.com"
# Clone your repo (you'll be prompted for GitHub token)
# !git clone https://github.com/tnsay/RAG-Powered-Chatbot.git
# !mkdir -p RAG-Powered-Chatbot/src
# get_ipython().system('cp /content/rag_pipeline.ipynb RAG-Powered-Chatbot/src/')


# In[ ]:


# from google.colab import files
# import shutil

# Set filename you want
notebook_name = "rag_pipeline.ipynb"

# This copies from the working notebook to /content/
# shutil.copyfile("/content/drive/MyDrive/Colab Notebooks/rag_pipeline.ipynb", f"/content/{notebook_name}")


# In[6]:


# # !jupyter nbconvert --to python src/rag_pipeline.ipynb
# get_ipython().system(' cd')

# get_ipython().system('dir c:\\PY\\RAG-Powered-Chatbot\\src')

