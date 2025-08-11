from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from llm import get_llm
from typing import Dict, List
from transformers import AutoTokenizer
import math


class RAGResearchAgent:
    def __init__(self, vector_db):
        self.retriever = vector_db.as_retriever(search_kwargs={"k": 10})
        self.tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        self.prompt = ChatPromptTemplate.from_template("""
        Answer this research question using ONLY the provided context.
        Cite sources as [Title, Page X].

        Question: {question}

        Context:
        {context}
        """)

    def _dynamic_num_ctx(self, context: str, question: str, answer_buffer: int = 512):
        token_count = len(self.tokenizer.encode(context + question))
        return min(math.ceil(token_count + answer_buffer), 65536)  # Cap at model max

    def query(self, question: str):
        # Retrieve docs
        docs = self.retriever.invoke(question)
        context = "\n\n".join(f"Paper Title : {doc.metadata['title']} \n Content : {doc.page_content}" for doc in docs)

        # Calculate optimal context window
        ctx_size = self._dynamic_num_ctx(context, question)

        # Get LLM with dynamic num_ctx
        print(f"Context length : {ctx_size}")
        llm = get_llm(num_ctx=ctx_size)

        # Run the model
        answer = llm.invoke(self.prompt.format(question=question, context=context))

        return {
            "answer": answer,
            "sources": list(
                set(f"{doc.metadata['title']} ({doc.metadata['year']})" for doc in docs)
            ),
            "num_ctx_used": ctx_size
        }
    
# Add at the end of rag_agent.py
from llm import get_llm
import os

class ReviewerAgent:
    """Agent that wraps RAGResearchAgent for literature review."""
    def __init__(self, vector_db):
        self.reviewer = RAGResearchAgent(vector_db)

    def run(self, question: str):
        return self.reviewer.query(question)


class PlannerAgent:
    """Agent that generates experiment plans from literature review."""
    def __init__(self):
        pass

    def plan_experiment(self, literature_summary: str) -> str:
        llm = get_llm(num_ctx=8192)
        prompt = f"""
You are a research planner. Based on the literature summary below, produce:
1) Objective, dataset, model, and loss/metric to test.
2) Evaluation plan (metrics and baselines).
3) Implementation notes and hyperparameters.

Literature Summary:
{literature_summary}
"""
        return llm.invoke(prompt)


class CoderAgent:
    """Agent that generates starter Python code for experiments."""
    def __init__(self):
        pass

    def generate_code(self, experiment_plan: str, out_dir: str = "generated_code"):
        llm = get_llm(num_ctx=8192)
        prompt = f"""
Given the experiment plan below, write a single Python script implementing it.
Include imports, requirements as comments at the top, and ensure it's runnable.

Experiment Plan:
{experiment_plan}
"""
        code = llm.invoke(prompt)

        os.makedirs(out_dir, exist_ok=True)
        file_path = os.path.join(out_dir, "experiment.py")
        with open(file_path, "w") as f:
            f.write(code)

        return {"experiment.py": code, "path": file_path}

