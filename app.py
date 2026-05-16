import streamlit as st
import tempfile
import os

from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

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

from test_questions import test_data


# =========================================================
# CONFIG
# =========================================================

st.set_page_config(page_title="Hybrid RAG + RAGAS", layout="wide")
st.title("⚡ Clean Hybrid RAG + RAGAS")


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "ragas_history" not in st.session_state:
    st.session_state.ragas_history = []


# =========================================================
# LLM (SAME FOR BOTH BUT SEPARATE PROMPTS)
# =========================================================

llm = ChatGroq(
    groq_api_key=st.secrets["GROQ_API_KEY"],
    model_name="llama-3.1-8b-instant"
)


# =========================================================
# EMBEDDINGS
# =========================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# =========================================================
# UPLOAD
# =========================================================

st.sidebar.title("Upload Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF / TXT / CSV / DOCX",
    type=["pdf", "txt", "csv", "docx"],
    accept_multiple_files=True
)


# =========================================================
# LOAD DOCS
# =========================================================

def load_document(file_path, file_type):

    if file_type == "pdf":
        return PyPDFLoader(file_path).load()
    elif file_type == "txt":
        return TextLoader(file_path).load()
    elif file_type == "csv":
        return CSVLoader(file_path).load()
    elif file_type == "docx":
        return Docx2txtLoader(file_path).load()
    return []


# =========================================================
# BUILD RETRIEVER
# =========================================================

@st.cache_resource
def build_retriever(files):

    docs = []

    for f in files:

        ext = f.name.split(".")[-1]

        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
            tmp.write(f.read())
            path = tmp.name

        loaded = load_document(path, ext)

        for d in loaded:
            d.metadata["source"] = f.name

        docs.extend(loaded)
        os.remove(path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80
    )

    split_docs = splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(split_docs, embeddings)

    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    bm25 = BM25Retriever.from_documents(split_docs)
    bm25.k = 2

    return EnsembleRetriever(
        retrievers=[faiss_retriever, bm25],
        weights=[0.8, 0.2]
    )


# =========================================================
# PROCESS FILES
# =========================================================

if uploaded_files:
    st.session_state.retriever = build_retriever(uploaded_files)
    st.success("Documents ready")


# =========================================================
# CHAT MODE (FIXED)
# =========================================================

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


user_question = st.chat_input("Ask your question...")

if user_question:

    st.session_state.messages.append({"role": "user", "content": user_question})

    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):

        if not st.session_state.retriever:
            st.warning("Upload documents first")

        else:

            docs = st.session_state.retriever.invoke(user_question)

            context = "\n\n".join([d.page_content[:600] for d in docs])

            # ✅ CHAT PROMPT ONLY (NO RAGAS MIXING)
            chat_prompt = f"""
You are a helpful assistant.

Answer ONLY using the context.

Context:
{context}

Question:
{user_question}

Answer:
"""

            response = llm.invoke(chat_prompt)
            answer = response.content.strip()

            st.write(answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })


# =========================================================
# RAGAS EVALUATION (SEPARATE ONLY)
# =========================================================

st.sidebar.title("RAGAS Evaluation")

if st.sidebar.button("Run RAGAS Evaluation"):

    if not st.session_state.retriever:
        st.sidebar.warning("Upload documents first")

    else:

        questions, answers, contexts, references = [], [], [], []

        for item in test_data:

            q = item["question"]
            r = item["reference"]

            docs = st.session_state.retriever.invoke(q)

            ctx = [d.page_content[:500] for d in docs]

            prompt = f"""
Answer using context only.

Context:
{chr(10).join(ctx)}

Question:
{q}

Answer:
"""

            res = llm.invoke(prompt)
            ans = res.content.strip()

            questions.append(q)
            answers.append(ans)
            contexts.append(ctx)
            references.append(r)

        dataset = Dataset.from_dict({
            "user_input": questions,
            "response": answers,
            "retrieved_contexts": contexts,
            "reference": references
        })

        result = evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall
            ],
            llm=llm,
            embeddings=embeddings
        )

        df = result.to_pandas()

        st.subheader("⚡ RAGAS Results")
        st.dataframe(df)

        st.subheader("Average Scores")

        st.write(f"Faithfulness: {df['faithfulness'].mean():.4f}")
        st.write(f"Answer Relevancy: {df['answer_relevancy'].mean():.4f}")
        st.write(f"Context Precision: {df['context_precision'].mean():.4f}")
        st.write(f"Context Recall: {df['context_recall'].mean():.4f}")

        # HISTORY STORE (ONLY RAGAS)
        for i in range(len(df)):
            st.session_state.ragas_history.append({
                "question": df["user_input"][i],
                "answer": df["response"][i],
                "faithfulness": df["faithfulness"][i],
                "answer_relevancy": df["answer_relevancy"][i],
                "context_precision": df["context_precision"][i],
                "context_recall": df["context_recall"][i],
            })


# =========================================================
# HISTORY (CLEAN SEPARATE DISPLAY)
# =========================================================

st.markdown("## 📚 RAGAS HISTORY")

for item in st.session_state.ragas_history[::-1]:

    st.markdown(f"**Q:** {item['question']}")
    st.markdown(f"**A:** {item['answer']}")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Faithfulness", item["faithfulness"])
    c2.metric("Relevancy", item["answer_relevancy"])
    c3.metric("Precision", item["context_precision"])
    c4.metric("Recall", item["context_recall"])

    st.divider()