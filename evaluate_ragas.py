from datasets import Dataset

from ragas import evaluate

from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

from langchain_groq import ChatGroq

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader
)

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS

from sentence_transformers import SentenceTransformer

from langchain_core.embeddings import Embeddings

from langchain_community.retrievers import BM25Retriever

from langchain_classic.retrievers import EnsembleRetriever

from test_questions import questions

import tempfile
import os

# ---------------------------------
# LLM
# ---------------------------------

llm = ChatGroq(
    groq_api_key="YOUR_GROQ_API_KEY",
    model_name="llama-3.1-8b-instant"
)

# ---------------------------------
# LOAD DOCUMENT
# ---------------------------------

file_path = "your_document.pdf"

loader = PyPDFLoader(file_path)

documents = loader.load()

# ---------------------------------
# SPLIT
# ---------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

split_docs = splitter.split_documents(documents)

# ---------------------------------
# EMBEDDINGS
# ---------------------------------

class CustomHFEmbeddings(Embeddings):

    def __init__(self):
        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode(text).tolist()

embeddings = CustomHFEmbeddings()

# ---------------------------------
# FAISS
# ---------------------------------

vectorstore = FAISS.from_documents(
    split_docs,
    embeddings
)

faiss_retriever = vectorstore.as_retriever(
    search_kwargs={"k": 2}
)

# ---------------------------------
# BM25
# ---------------------------------

bm25_retriever = BM25Retriever.from_documents(
    split_docs
)

bm25_retriever.k = 4

# ---------------------------------
# HYBRID RETRIEVER
# ---------------------------------

retriever = EnsembleRetriever(
    retrievers=[
        faiss_retriever,
        bm25_retriever
    ],
    weights=[0.7, 0.3]
)

# ---------------------------------
# CREATE EVALUATION DATA
# ---------------------------------

answers = []
contexts = []

for question in questions:

    docs = retriever.invoke(question)

    context_texts = [
        doc.page_content
        for doc in docs
    ]

    context = "\n".join(context_texts)

    prompt = f"""
Answer ONLY from context.

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)

    answer = response.content

    answers.append(answer)

    contexts.append(context_texts)

# ---------------------------------
# DATASET
# ---------------------------------

dataset = Dataset.from_dict({
    "question": questions,
    "answer": answers,
    "contexts": contexts
})

# ---------------------------------
# RAGAS EVALUATION
# ---------------------------------

result = evaluate(
    dataset=dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ]
)

print(result)