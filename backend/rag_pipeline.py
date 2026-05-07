import os
#from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
#from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama


# Directory to store vector databases
DB_DIR = "vector_stores"

if not os.path.exists(DB_DIR):
    os.makedirs(DB_DIR)


# 1. Load PDF
def load_pdf(file_path):
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()

    print("PAGES LOADED:", len(docs))

    if docs:
        print("FIRST PAGE CONTENT:")
        print(docs[0].page_content[:1000])

    return docs


# 2. Split into chunks
def split_docs(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_documents(documents)


# 3. Create embeddings
def create_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-mpnet-base-v2"
    )


# 4. Store FAISS database
def store_in_faiss(docs, embeddings, session_id):
    db = FAISS.from_documents(docs, embeddings)

    save_path = os.path.join(DB_DIR, session_id)
    db.save_local(save_path)

    return db


# 5. Load existing FAISS DB
def load_faiss(session_id, embeddings):
    save_path = os.path.join(DB_DIR, session_id)

    if os.path.exists(save_path):
        return FAISS.load_local(
            save_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

    return None


# 6. Ask question
def ask_question(db, query):
    try:
        retriever = db.as_retriever(search_kwargs={"k": 4})

        # Special handling for summary/overview type questions
        if "summary" in query.lower() or "overview" in query.lower():
            docs = db.similarity_search("", k=4)
        else:
            docs = retriever.invoke(query)

        print("RETRIEVED DOCS:", len(docs))
        for i, doc in enumerate(docs):
            print(f"\nDOC {i+1}:")
            print(doc.page_content[:500])
        # Filter weak chunks
        # docs = [
        #     doc for doc in docs
        #     if len(doc.page_content.strip()) > 50
        # ]

        # Extract sources
        sources = sorted(
            list(
                set([
                    doc.metadata.get("page", 0) + 1
                    for doc in docs
                ])
            )
        )

        # Build context
        context_text = "\n\n".join([
            f"Page {doc.metadata.get('page', 0) + 1}:\n{doc.page_content}"
            for doc in docs
        ])

        print("CONTEXT PREVIEW:\n", context_text[:1000])

        if not context_text.strip():
            return (
                "I couldn't find any relevant information in the document to answer that question.",
                []
            )

        llm = ChatOllama(
        model="phi",
        temperature=0
        )

        prompt = f"""
You are a document question-answering assistant.

IMPORTANT RULES:
1. Answer ONLY using the provided context
2. Do NOT add extra information
3. Do NOT make assumptions
4. Keep the answer short and precise
5. Do NOT expand abbreviations unless explicitly written
6. Do NOT invent facts
7. If the answer is not in the context, reply exactly:
   "Not found in document"

CONTEXT:
{context_text}

QUESTION:
{query}

ANSWER:
"""

        response = llm.invoke(prompt)

        answer = response.content.strip()

        return answer, sources

    except Exception as e:
        print("ERROR IN ask_question():", str(e))

        import traceback
        traceback.print_exc()

        return (
            f"Error generating answer: {str(e)}",
            []
        )

