from pathlib import Path

class RAGSystem:
    def __init__(
        self,
        top_k: int = 10,
        final_k: int = 5,
    ):
        self.top_k = top_k
        self.final_k = final_k
        pass

    def query(
        self,
        query: str,
        use_hybrid: bool,
        use_query_expansion: bool,
    ):
        pass
        # debug_info = {}

        # search_query = query
        # if use_query_expansion:
        #     expanded_queries = query_expander.expand_with_synonms(query)
        #     search_query = " ".join(expanded_queries[:3])
        #     debug_info["expanded_queries"] = expanded_queries

        # retrieval_results = retriever.retrieve(
        #     query=search_query,
        #     top_k=self.top_k,
        #     final_k=self.final_k,
        #     use_hybrid=use_hybrid,
        # )

    def parse(
            self,
            pdf_dir: str,
            output_dir: str,
    ):
        result = {}
        pdf_files = Path(pdf_dir).glob("*.pdf")
