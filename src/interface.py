
def chat_interface(
        query: str,
        history: list,
        use_hybrid: bool,
        use_expansion: bool,
) -> tuple[str, list, str]:
    if not query.strip():
        return "", history, ""

    answer, citations, debug_info = rag_query(
        query=query,
        use_hybrid=use_hybrid,
        use_query_expansion=use_expansion,
    )

    history.append((query, answer))

    return "", history, debug_info
