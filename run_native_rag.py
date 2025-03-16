import re

from exa_search_main import (
    extract_snippet_with_context,
    exa_web_search,
    extract_relevant_info,
    extract_snippet_with_context,
    fetch_page_content,
)
from vllm_inference import chat_without_model


def native_rag_main(query, model_url):
    url = "https://api.exa.ai/search"
    # query = "刘翔获得了多少次冠军"
    results = exa_web_search(query, url, num_result=10)
    top_k = 10
    max_doc_len = 3000
    print(results)

    relevant_info = extract_relevant_info(results)[:top_k]

    # Collect all unique URLs to fetch
    unique_urls = set()
    url_snippets_map = {}

    for info in relevant_info:
        url = info['url']
        snippet = info.get('snippet', "")
        unique_urls.add(url)
        url_snippets_map[url] = query

    # Determine which URLs need to be fetched
    urls_to_fetch = [url for url in unique_urls]

    print(f"Fetching {len(urls_to_fetch)} unique URLs...")
    fetched_contents = fetch_page_content(
        urls_to_fetch,
        snippets=url_snippets_map
    )

    formatted_documents = ""
    for i, doc_info in enumerate(relevant_info):
        url = doc_info['url']
        # snippet = doc_info.get('snippet', "")
        snippet = query
        raw_context = fetched_contents[url]
        success, context = extract_snippet_with_context(raw_context, snippet, context_chars=max_doc_len)
        if success:
            context = context
        else:
            context = raw_context[:2 * max_doc_len]

        # Clean snippet from HTML tags if any
        clean_snippet = re.sub('<[^<]+?>', '', snippet)  # Removes HTML tags

        formatted_documents += f"**Document {i + 1}:**\n"
        formatted_documents += f"**Title:** {doc_info.get('title', '')}\n"
        formatted_documents += f"**URL:** {url}\n"
        formatted_documents += f"**Snippet:** {clean_snippet}\n"
        formatted_documents += f"**Content:** {context}\n\n"

    # Construct the instruction with documents and question
    # instruction = get_naive_rag_instruction(question, formatted_documents)
    print(formatted_documents)

    instruction = (
        "You are a knowledgeable assistant that uses the provided documents to answer the user's question.\n\n"
        "Question:\n"
        f"{query}\n"
        "Documents:\n"
        f"{formatted_documents}\n"
    )

    user_prompt = (
        'Please answer the following question. You should think step by step to solve it.\n\n'
        'Provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'
        f'Question:\n{query}\n\n'
    )

    full_prompt = instruction + "\n\n" + user_prompt

    response = chat_without_model(full_prompt, model_url)
    return response


if __name__ == '__main__':
    model_url = "http://192.168.16.6:8000/v1"
    message = "刘翔获得了多少个冠军？"
    message = "《哪吒之魔童闹海》目前的票房是多少？位于全球票房第几？"
    message = "《哪吒之魔童闹海》的目前票房是多少？"
    response = native_rag_main(message, model_url)
    print(response)
