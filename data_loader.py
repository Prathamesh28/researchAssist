import requests
import fitz  # PyMuPDF
from scholarly import scholarly
from arxiv import Search, SortCriterion
from typing import List, Dict

class MultiSourcePaperLoader:
    def __init__(self, semantic_api_key: str = None):
        self.semantic_base_url = "https://api.semanticscholar.org/graph/v1"
        self.semantic_api_key = semantic_api_key

    from urllib.parse import urlparse

    def get_pdf_url(self, paper_url: str) -> str:
        """
        Convert a paper HTML page URL to a direct PDF URL for known publishers.
        Supports:
        - PMLR Proceedings (proceedings.mlr.press)
        - arXiv (arxiv.org)
        - CVF Open Access (openaccess.thecvf.com)
        """
        if "proceedings.mlr.press" in paper_url:
            # Extract the volume and paper_id
            parts = paper_url.rstrip("/").split("/")
            volume = parts[-2]   # e.g., v119
            paper_id_html = parts[-1]  # e.g., wang20t.html
            paper_id = paper_id_html.replace(".html", "")
            return f"https://proceedings.mlr.press/{volume}/{paper_id}/{paper_id}.pdf"
        
        elif "arxiv.org/abs/" in paper_url:
            # Example: https://arxiv.org/abs/1706.01061 -> /pdf/1706.01061.pdf
            return paper_url.replace("abs", "pdf") + ".pdf"
        
        elif "openaccess.thecvf.com" in paper_url:
            # Example:
            # https://openaccess.thecvf.com/content_cvpr_2016/html/Cheng_...html
            # -> https://openaccess.thecvf.com/content_cvpr_2016/papers/Cheng_...pdf
            return paper_url.replace(".html", ".pdf").replace("/html/", "/papers/")
        
        else:
            return None

    # ---------------- SEMANTIC SCHOLAR ----------------
    def fetch_from_semantic_scholar(self, query: str, max_results=3) -> List[Dict]:
        headers = {}
        if self.semantic_api_key:
            headers["x-api-key"] = self.semantic_api_key
        
        url = f"{self.semantic_base_url}/paper/search"
        params = {
            "query": query,
            "limit": max_results,
            "fields": "title,url,abstract,authors,year,openAccessPdf"
        }

        try:
            res = requests.get(url, headers=headers, params=params, timeout=15)
            res.raise_for_status()
            data = res.json()
            results = []
            for paper in data.get("data", []):
                pdf_url = paper.get("openAccessPdf", {}).get("url")
                if not pdf_url:
                    continue
                results.append({
                    "title": paper["title"],
                    "abstract": paper.get("abstract"),
                    "authors": [a["name"] for a in paper.get("authors", [])],
                    "year": paper.get("year"),
                    "url": paper.get("url"),
                    "pdf_url": pdf_url
                })
            return results
        except Exception as e:
            print(f"[Semantic Scholar Error] {e}")
            return []

    # ---------------- ARXIV ----------------
    def fetch_from_arxiv(self, query: str, max_results=3) -> List[Dict]:
        try:
            search = Search(query=query, max_results=max_results, sort_by=SortCriterion.Relevance)
            results = []
            for r in search.results():
                if not r.pdf_url:
                    continue

                results.append({
                    "title": r.title,
                    "abstract": r.summary,
                    "authors": [a.name for a in r.authors],
                    "year": r.published.year,
                    "url": r.entry_id,
                    "pdf_url": r.pdf_url
                })
            return results
        except Exception as e:
            print(f"[arXiv Error] {e}")
            return []

    # ---------------- GOOGLE SCHOLAR ----------------
    def fetch_from_google_scholar(self, query: str, max_results=3) -> List[Dict]:
        try:
            search_query = scholarly.search_pubs(query)
            results = []
            for i, paper in enumerate(search_query):
                if i >= max_results:
                    break

                pdf_url = self.get_pdf_url(paper.get("pub_url", ""))
                if not pdf_url:  # Skip if PDF URL can't be determined
                    continue
                results.append({
                    "title": paper.get("bib", {}).get("title"),
                    "abstract": paper.get("bib", {}).get("abstract"),
                    "authors": paper.get("bib", {}).get("author"),
                    "year": paper.get("bib", {}).get("pub_year"),
                    "url": paper.get("pub_url"),
                    "pdf_url": pdf_url  # Needs manual check
                })

            
            return results
        except Exception as e:
            print(f"[Google Scholar Error] {e}")
            return []

    # ---------------- PDF TEXT EXTRACTION ----------------
    def _extract_text_from_pdf(self, pdf_url: str) -> str:
        try:
            res = requests.get(pdf_url, timeout=20)
            res.raise_for_status()
            with fitz.open(stream=res.content, filetype="pdf") as doc:
                return "\n".join(page.get_text() for page in doc)
        except Exception as e:
            print(f"[PDF Extraction Error] {e}")
            return ""

    # ---------------- MASTER FETCH METHOD ----------------
    def fetch_papers(self, query: str, max_results=3) -> List[Dict]:
        results = []

        # Try Semantic Scholar
        results.extend(self.fetch_from_semantic_scholar(query, max_results))
        # If insufficient, try arXiv
        # print(len(results))
        if len(results) < max_results:
            results.extend(self.fetch_from_arxiv(query, max_results))
        # print(len(results))
        
        # If still insufficient, try Google Scholar
        if len(results) < max_results:
            results.extend(self.fetch_from_google_scholar(query, max_results))


        # Remove duplicates based on title
        seen_titles = set()
        unique_results = []
        for paper in results:
            if paper["title"] and paper["title"].lower() not in seen_titles:
                seen_titles.add(paper["title"].lower())
                unique_results.append(paper)

        # Fetch PDF text if available
        for paper in unique_results:
            if paper.get("pdf_url"):
                paper["full_text"] = self._extract_text_from_pdf(paper["pdf_url"])
            else:
                paper["full_text"] = ""

        return unique_results

# Add at the end of data_loader.py
class RetrieverAgent:
    """Agent that wraps MultiSourcePaperLoader for paper retrieval."""
    def __init__(self, semantic_api_key: str = None):
        self.loader = MultiSourcePaperLoader(semantic_api_key=semantic_api_key)

    def run(self, query: str, max_papers: int = 10):
        return self.loader.fetch_papers(query, max_results=max_papers)


# ---------------- Example usage ----------------
# if __name__ == "__main__":
#     loader = MultiSourcePaperLoader(semantic_api_key=None)  # Add key if available
#     papers = loader.fetch_papers("What are the loss functions used in CNN modelling for face detection?", max_results=10)

#     for p in papers:
#         print(f"Title: {p['title']}")
#         print(f"Authors: {p['authors']}")
#         print(f"Year: {p['year']}")
#         print(f"URL: {p['url']}")
#         print(f"Abstract: {p['abstract'][:200] if p['abstract'] else 'N/A'}\n")
#         print(f"Full Text: {p['full_text'][:100]}")
