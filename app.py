import streamlit as st
from data_loader import MultiSourcePaperLoader
from vectordb import ResearchVectorDB
from rag_agent import RAGResearchAgent
from config_loader import load_config
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

config = load_config()

def main():
    st.title("Academic Research Assistant")
    
    query = st.text_input("Enter your research question:")
    # query = "What are the loss functions used in CNN modelling for face detection?"
    if not query:
        return
    
    with st.spinner("🔍 Searching arXiv..."):
        loader = MultiSourcePaperLoader(semantic_api_key=None) 
        papers = loader.fetch_papers(query)[:config["processing"]["max_papers"]]
        
        if not papers:
            st.error("No papers found. Try a different query.")
            return
        print(f"Number of papers : {len(papers)}")
        for p in papers:
            print(f"Title: {p['title']}")
            print(f"Authors: {p['authors']}")
            print(f"Year: {p['year']}")
            print(f"URL: {p['pdf_url']}")
            print(f"Full Text: {p['full_text'][:100]}")

        import os

        # if os.path.exists("vector_store"):
        #     embeddings = HuggingFaceEmbeddings(
        #         model_name="sentence-transformers/all-MiniLM-L6-v2",  # Fast & small
        #         model_kwargs={"device": "cpu"},  # or "mps" for Apple Silicon
        #         encode_kwargs={"normalize_embeddings": True}
        #     )
        #     vector_db = FAISS.load_local("vector_store", embeddings, allow_dangerous_deserialization=True)
        # else:
        vector_db = ResearchVectorDB().index_papers(papers)
            # vector_db.save_local("vector_store")

        print("Vector store saved and loaded.")
        agent = RAGResearchAgent(vector_db)
        
        with st.spinner("📚 Analyzing papers..."):
            response = agent.query(query)
            print(response) 
            
        st.markdown(response["answer"])
        st.divider()
        st.subheader("📚 Sources")
        for source in response["sources"]:
            st.caption(f"- {source}")

if __name__ == "__main__":
    main()