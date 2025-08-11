from data_loader import RetrieverAgent
from vectordb import IndexerAgent
from rag_agent import ReviewerAgent, PlannerAgent, CoderAgent
from config_loader import load_config
import streamlit as st

config = load_config()

def main():
    st.title("Multi-Agent Academic Research Assistant")

    query = st.text_input("Enter your research question:")
    max_papers = st.slider("Max papers to fetch", min_value=1, max_value=30, value=config["processing"]["max_papers"])

    if not query:
        return

    if st.button("Run Pipeline"):
        # 1. Retrieve papers
        with st.spinner("🔍 Retrieving papers..."):
            retriever = RetrieverAgent()
            papers = retriever.run(query, max_papers=max_papers)
            st.success(f"Retrieved {len(papers)} papers.")
            for p in papers:
                st.caption(f"{p.get('title')} ({p.get('year')}) - {p.get('pdf_url')}")

        # 2. Index papers
        with st.spinner("📚 Indexing papers..."):
            indexer = IndexerAgent()
            vector_db = indexer.run(papers)
            st.success("Indexing complete.")

        # 3. Literature review
        with st.spinner("🧠 Reviewing literature..."):
            reviewer = ReviewerAgent(vector_db)
            review = reviewer.run(query)
            st.subheader("Literature Review")
            st.write(review["answer"])
            st.caption("Sources:")
            for s in review["sources"]:
                st.caption(s)

        # 4. Experiment planning
        with st.spinner("📝 Planning experiment..."):
            planner = PlannerAgent()
            plan = planner.plan_experiment(review["answer"])
            st.subheader("Experiment Plan")
            st.write(plan)

        # 5. Code generation
        with st.spinner("💻 Generating starter code..."):
            coder = CoderAgent()
            code_files = coder.generate_code(plan, out_dir=config.get("processing", {}).get("out_dir", "generated_code"))
            st.subheader("Generated Code")
            st.code(code_files["experiment.py"], language="python")
            st.success(f"Code saved to: {code_files['path']}")

if __name__ == "__main__":
    main()
