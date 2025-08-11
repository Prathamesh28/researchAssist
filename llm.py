from langchain_community.llms import Ollama
from config_loader import load_config

config = load_config()

def get_llm(num_ctx=config["ollama"]["num_ctx"]):
    return Ollama(
        base_url=config["ollama"]["base_url"],
        model=config["ollama"]["model_name"],
        temperature=config["ollama"]["temperature"],
        num_ctx=num_ctx,
        top_k=20
    )