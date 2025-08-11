
# ResearchAssist – AI-Powered Literature Review Assistant

## 📌 Overview
**ResearchAssist** is an AI-powered academic assistant that automates literature review by:
- Searching across **multiple research sources** (Semantic Scholar, arXiv, Google Scholar)  
- Extracting and processing **full-text PDFs** of research papers  
- Building a **local vector database** for semantic search  
- Using **RAG (Retrieval-Augmented Generation)** with LLaMA 3 to answer research queries with **cited sources**  

This tool drastically reduces the time spent on finding, reading, and summarizing research papers, enabling faster **idea validation** and **knowledge discovery**.

---

## 🚀 Features
- 🔍 **Multi-source Search** – Fetch papers from **arXiv**, **Semantic Scholar**, and **Google Scholar**  
- 📄 **Full-Text Extraction** – Automatically download and parse research PDFs  
- 📚 **Semantic Search** – Build a **FAISS**-powered local knowledge base  
- 🤖 **AI-Powered Q&A** – Retrieve relevant context and answer queries with citations using **Ollama + LLaMA**  
- ⚡ **Local & Free** – Fully offline processing after data is downloaded  

---

## 🛠 Tech Stack
- **Python 3.10+**
- **LangChain** – Vector store, RAG pipeline  
- **FAISS** – Efficient semantic search  
- **SentenceTransformers** – Paper embeddings (`all-MiniLM-L6-v2`)  
- **Ollama** – Local LLaMA 3 inference  
- **Streamlit** – Interactive UI  
- **PyMuPDF (fitz)** – PDF parsing  
- **scholarly / arxiv API / Semantic Scholar API** – Paper search  

---

## 📂 Project Structure
\`\`\`
researchAssist/
│── app.py                # Streamlit UI
│── data_loader.py        # Fetch papers from multiple sources
│── vectordb.py           # FAISS vector store builder
│── rag_agent.py          # RAG pipeline with LLaMA
│── llm.py                # LLaMA model loader
│── config_loader.py      # Config management
│── requirements.txt
│── README.md
\`\`\`

---

## ⚙️ Installation
\`\`\`bash
# Clone repository
git clone https://github.com/Prathamesh28/researchAssist.git
cd researchAssist

# Install dependencies
pip install -r requirements.txt

# Run Ollama server locally
ollama serve

# Start the Streamlit app
streamlit run app.py
\`\`\`

---

## 💡 Usage
1. Enter a **research question** in the UI  
2. The system searches for relevant papers and downloads PDFs  
3. Papers are split into **chunks**, embedded, and indexed in FAISS  
4. AI retrieves relevant context and answers your question with **citations**  

---

## 📈 Impact
- Saves **hours of manual reading** by instantly summarizing relevant papers  
- Improves **research quality** with context-backed AI answers  
- Scales to **any research domain** without retraining  

---

## 📜 License
MIT License – Free to use and modify.
