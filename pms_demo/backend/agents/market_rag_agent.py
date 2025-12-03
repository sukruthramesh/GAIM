from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain import PromptTemplate
from pathlib import Path
import os
import glob
import json

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "rag_fusion.tpl"
with open(PROMPT_PATH, "r") as f:
    RAG_PROMPT = f.read()

CHROMA_DIR = "./.chromadb"

def ingest_documents(doc_folder: str, collection_name="pms_docs"):
    """
    Ingests text/PDF files into Chroma. For demo, assume docs are plain text or small PDFs processed elsewhere.
    """
    embeddings = OpenAIEmbeddings()
    texts = []
    for filepath in glob.glob(os.path.join(doc_folder, "*")):
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        texts.append(Document(page_content=content, metadata={"source": filepath}))
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(texts)
    vect = Chroma.from_documents(docs, embeddings, collection_name=collection_name, persist_directory=CHROMA_DIR)
    vect.persist()
    return vect

def rag_query_summarize(query: str, top_k: int = 5, collection_name="pms_docs", llm_model="gpt-4o-mini"):
    embeddings = OpenAIEmbeddings()
    vect = Chroma(collection_name=collection_name, persist_directory=CHROMA_DIR, embedding_function=embeddings)
    docs = vect.similarity_search(query, k=top_k)
    summaries = []
    for d in docs:
        summaries.append({"page_content": d.page_content[:1200], "source": d.metadata.get("source","")})
    # Build prompt
    retrieved_summaries = "\n\n".join([f"SOURCE: {s['source']}\n{ s['page_content'] }" for s in summaries])
    prompt = RAG_PROMPT.replace("{retrieved_summaries}", retrieved_summaries)
    llm = ChatOpenAI(temperature=0.0, model=llm_model)
    chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(prompt))
    raw = chain.run({})
    try:
        parsed = json.loads(raw)
    except Exception:
        # fallback simple defaults
        parsed = {
            "mu": {"domestic_equity": 0.06, "domestic_bonds": 0.02, "gold": 0.03, "intl_equity": 0.05},
            "cov_scale": 1.0,
            "sources": [d.metadata.get("source","") for d in docs]
        }
    return parsed
