from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config_loader import load_config
import re

config = load_config()

class ResearchVectorDB:
    def __init__(self):

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",  # Fast & small
            model_kwargs={"device": "cpu"},  # or "mps" for Apple Silicon
            encode_kwargs={"normalize_embeddings": True}
        )
        
        # Convert string separators to actual regex patterns
        import re

        separators = [
            re.compile(s) if s.startswith("(") or "\\" in s else s
            for s in config["processing"]["separators"]
        ]


        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config["processing"]["chunk_size"],
            chunk_overlap=config["processing"]["chunk_overlap"],
            separators=separators
        )


    
    def index_papers(self, papers):
        docs = []
        for paper in papers:
            text = paper.get("full_text", "")
            if not text.strip():
                print(f"⚠️ Skipping paper '{paper.get('title', 'Untitled')}' — no full text.")
                continue
            
            chunks = self.text_splitter.split_text(text)
            if not chunks:
                print(f"⚠️ No chunks created for '{paper.get('title', 'Untitled')}'.")
                continue

            for i, chunk in enumerate(chunks):
                docs.append(Document(
                    page_content=chunk,
                    metadata={
                        "title": paper.get("title", ""),
                        "authors": ", ".join(paper.get("authors", [])),
                        "year": paper.get("year", ""),
                        "page": f"{i+1}/{len(chunks)}"
                    }
                ))

        if not docs:
            raise ValueError("No documents to index — check paper loading and text splitting.")

        print(f"✅ Prepared {len(docs)} chunks for FAISS indexing.")
        print("🔹 Example chunk:", docs[0].page_content[:200])
        
        return FAISS.from_documents(docs, self.embeddings)


# Add at the end of vectordb.py
class IndexerAgent:
    """Agent that wraps ResearchVectorDB for indexing papers."""
    def __init__(self):
        self.indexer = ResearchVectorDB()

    def run(self, papers):
        return self.indexer.index_papers(papers)
