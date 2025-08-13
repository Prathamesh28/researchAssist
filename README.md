# ResearchAssist – AI-Powered Literature Review Assistant

## 📌 Overview
**ResearchAssist** is an AI-powered academic assistant that automates literature review by:
- Searching across **multiple research sources** (Semantic Scholar, arXiv, Google Scholar)  
- Extracting and processing **full-text PDFs** of research papers  
- Building a **local vector database** for semantic search  
- Using **RAG (Retrieval-Augmented Generation)** with LLaMA 3 to answer research queries with **cited sources**  

This tool drastically reduces the time spent on finding, reading, and summarizing research papers, enabling faster **idea validation** and **knowledge discovery**.

---
## Demo
[Click here for DEMO
]([https://prathamesh28.github.io/researchAssist/](https://prathamesh28.github.io/researchAssist/researchProjectDemo.mp4))
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
- **LangChain** – Document processing & retrieval  
- **FAISS** – Vector database for semantic search  
- **HuggingFace Embeddings** – Sentence-transformers for text embeddings  
- **Ollama + LLaMA 3** – Local LLM for Q&A  
- **PyMuPDF (fitz)** – PDF parsing  
- **Scholarly, arxiv, Semantic Scholar API** – Paper fetching  

---

## 📂 Project Structure
```
researchAssist/
│── app.py                 # Streamlit frontend for user interaction
│── data_loader.py         # Multi-source research paper fetching
│── vectordb.py            # FAISS vector database creation & indexing
│── rag_agent.py           # RAG pipeline for AI-powered Q&A
│── llm.py                  # Ollama LLaMA model integration
│── config_loader.py       # Loads configuration settings
│── requirements.txt       # Python dependencies
│── README.md              # Project documentation
```

---

## ⚙️ Installation

1️⃣ **Clone the Repository**
```bash
git clone https://github.com/Prathamesh28/researchAssist.git
cd researchAssist
```

2️⃣ **Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate    # On Windows
```

3️⃣ **Install Dependencies**
```bash
pip install -r requirements.txt
```

4️⃣ **Install Ollama & LLaMA**
- Download Ollama from [https://ollama.ai](https://ollama.ai)  
- Pull LLaMA 3 model:
```bash
ollama pull llama3
```

---

## ▶️ Running the App
```bash
streamlit run app.py
```

---

## 💡 Example Queries
- *"What are the loss functions used in CNN modelling for face detection?"*  
- *"Recent advancements in few-shot learning for NLP"*  
- *"How diffusion models are applied in image segmentation"*  

---

## 📜 License
MIT License © 2025 Prathamesh Wagh
