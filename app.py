import ssl
import certifi
import os
import streamlit as st

# SSL FIX
ssl._create_default_https_context = ssl.create_default_context(
    cafile=certifi.where()
)

ssl._create_default_https_context = ssl._create_unverified_context

os.environ["CURL_CA_BUNDLE"] = ""
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"

# LANGCHAIN IMPORTS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder


# PAGE CONFIG
st.set_page_config(
    page_title="DMRC HR Chatbot",
    page_icon="logo.png",
    layout="wide"
)

# HEADER
col1, col2 = st.columns([0.4, 4])

with col1:
    st.image("logo.png", width=80)

with col2:
    st.markdown(
        "<h1 style='margin-top:15px;'>DMRC HR Chatbot</h1>",
        unsafe_allow_html=True
    )

st.write("Ask questions related to DMRC HR policies.")


# GROQ API KEY
GROQ_API_KEY = "API_KEY"


# LOAD EMBEDDING MODEL
@st.cache_resource
def load_embedding():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5"
    )

embedding_model = load_embedding()


# LOAD FAISS DATABASE
@st.cache_resource
def load_db():
    return FAISS.load_local(
        "faiss_index",
        embedding_model,
        allow_dangerous_deserialization=True
    )

db = load_db()

# LOAD ALL DOCUMENTS FROM FAISS
all_docs = db.similarity_search("", k=200)

# BM25 RETRIEVER
bm25_retriever = BM25Retriever.from_documents(all_docs)
bm25_retriever.k = 6

# LOAD GROQ MODEL
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile"
)

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# CHAT MEMORY
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# DISPLAY CHAT HISTORY
for message in st.session_state.chat_history:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# USER INPUT
query = st.chat_input("Ask your question to HR...")

def rerank_documents(query, docs):

    pairs = []

    for doc in docs:
        pairs.append([query, doc.page_content])

    scores = reranker.predict(pairs)

    scored_docs = list(zip(scores, docs))

    scored_docs.sort(
        key=lambda x: x[0],
        reverse=True
    )

    reranked_docs = [doc for score, doc in scored_docs]

    return reranked_docs[:6]



# MAIN CHATBOT LOGIC
if query:

    # SHOW USER MESSAGE
    with st.chat_message("user"):
        st.markdown(query)

    # SMART RETRIEVAL
    vector_retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 12}
)

    # Increase BM25 depth
    bm25_retriever.k = 10

    # Hybrid retriever
    ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6]
)

    docs = ensemble_retriever.invoke(query)
    docs = rerank_documents(query, docs)

    # CREATE CONTEXT
    context = "\n\n".join([
    f"[Source: {doc.metadata.get('source', 'Unknown')} | Page: {doc.metadata.get('page', 'N/A')}]\n{doc.page_content}"
    for doc in docs
])
        
    # CREATE HISTORY STRING
    history = ""

    for msg in st.session_state.chat_history:
        history += f"{msg['role']}: {msg['content']}\n"    

    prompt = f"""
You are a DMRC HR policy analyst.

Answer ONLY using the provided context.

Instructions:
- Extract exact rules, conditions, and clauses
- Combine multiple clauses if needed
- Do NOT generalize beyond context
- Do NOT give generic HR explanations
- Base EVERY statement strictly on provided context
- If answer not found, say: "Not found in policy documents"
For every conclusion, support it with policy context.
Avoid vague or general statements.

For the given question, provide:

1. Direct Answer
2. Relevant Policy Rules (bullet points)
3. Conditions / Exceptions
4. Practical Interpretation (HR decision view)

Context:
{context}

Question:
{query}

Answer:
"""

    # GENERATE RESPONSE
    response = llm.invoke(prompt)

    answer = response.content

    # SHOW AI RESPONSE
    with st.chat_message("assistant"):
        st.markdown(answer)

    # SAVE TO MEMORY
    st.session_state.chat_history.append({
        "role": "user",
        "content": query
    })

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer
    })

    # SOURCES SECTION
    st.subheader("Policy References")

    for i, doc in enumerate(docs):

        source_text = doc.page_content[:500]

        # SOURCE SUMMARIZATION PROMPT
        source_prompt = f"""
You are an HR policy summarizer.

Convert the following HR policy text into:
- short
- professional
- intelligent
- easy-to-understand language

Avoid copying exact lines.

Policy:
{source_text}
"""

        source_response = llm.invoke(source_prompt)

        with st.expander(f"Reference {i+1}"):

            st.markdown(source_response.content)

            # PAGE NUMBER
            if "page" in doc.metadata:
                st.write("Page:", doc.metadata["page"])

            # SOURCE NAME
            if "source" in doc.metadata:
                st.write("Source:", doc.metadata["source"])

            # OPTIONAL PDF BUTTON
            pdf_path = "HR Compendium 2025.pdf"

            if os.path.exists(pdf_path):

                with open(pdf_path, "rb") as file:

                    st.download_button(
                        label="Open Source PDF",
                        data=file,
                        file_name="HR_Compendium_2025.pdf",
                        mime="application/pdf",
                        key=f"pdf_{i}"
                    )
