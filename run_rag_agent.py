import os
import re
import time
import json
from pprint import pprint
from typing import Optional

from exa_search_main import (
    extract_snippet_with_context,
    exa_web_search,
    extract_relevant_info,
    extract_snippet_with_context,
    fetch_page_content,
)
from vllm_inference import chat_without_model

BEGIN_SEARCH_QUERY = "<|begin_search_query|>"
END_SEARCH_QUERY = "<|end_search_query|>"
BEGIN_SEARCH_RESULT = "<|begin_search_result|>"
END_SEARCH_RESULT = "<|end_search_result|>"
BEGIN_URL = "<|begin_url|>"
END_URL = "<|end_url|>"
BEGIN_FULL_PAGE = "<|begin_full_page|>"
END_FULL_PAGE = "<|end_full_page|>"
MAX_SEARCH_LIMIT = 5
MAX_URL_FETCH = 100
MAX_TURN = 10
max_tokens = 32768
top_k = 10


def run_generation(sequences_needing_generation, model_url):
    print(sequences_needing_generation[0]["prompt"])
    result = chat_without_model(sequences_needing_generation[0]["prompt"], model_url)
    return [result]


# Function to extract text between two tags
def extract_between(text: str, start_tag: str, end_tag: str) -> Optional[str]:
    pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
    matches = re.findall(pattern, text, flags=re.DOTALL)
    print(matches)
    if matches:
        return matches[-1].strip()
    return None


def save_caches(url_cache_path, url_cache):
    with open(url_cache_path, 'w', encoding='utf-8') as f:
        json.dump(url_cache, f, ensure_ascii=False, indent=2)


def rag_agent_main(query, model_url):
    cache_dir = './cache'
    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)

    url_cache_path = os.path.join(cache_dir, 'url_cache.json')

    if os.path.exists(url_cache_path):
        with open(url_cache_path, 'r', encoding='utf-8') as f:
            url_cache = json.load(f)
    else:
        url_cache = {}

    url = "https://api.exa.ai/search"

    instruction = (
        "You are a reasoning assistant with the ability to perform web searches and retrieve webpage content to help "
        "you answer the user’s question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will call the web search API with your query and return the search results to you in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n"
        "  The search results will contain a list of webpages with titles, URLs, and snippets (but not full content).\n\n"
        "- After receiving the search results, if you need more detailed information from one or more specific URLs, write <|begin_url|> url1, url2, ... <|end_url|>.\n"
        "  The system will fetch the full page content of those URLs and return it to you as <|begin_full_page|> ...full page content... <|end_full_page|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n"
        f"You can fetch up to {MAX_URL_FETCH} URLs for detailed information.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"Who got the first Nobel Prize in Physics?\"\n"
        "Assistant thinking steps:\n"
        "- I need to find out who was awarded the first Nobel Prize in Physics.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>first Nobel Prize in Physics winner<|end_search_query|>\n\n"
        "(System returns search results)\n\n"
        "Assistant:\n"
        "<|begin_search_result|> ...search results without full page... <|end_search_result|>\n\n"
        "Assistant thinks: The search results mention several URLs. I want full details from one of them.\n\n"
        "Assistant:\n"
        "<|begin_url|>http://example.com/first_nobel_physics.html<|end_url|>\n\n"
        "(System returns full page content)\n\n"
        "Assistant:\n"
        "<|begin_full_page|> ...full page content... <|end_full_page|>\n\n"
        "Now the assistant has enough info and can continue reasoning.\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- Use <|begin_url|> to request full page content and end with <|end_url|>.\n"
        "- When done retrieving information, continue your reasoning.\n\n"
    )

    user_prompt = (
        'Please answer the following question. You should think step by step to solve it.\n\n'
        'Provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'
        f'Question:\n{query}\n\n'
    )

    full_prompt = instruction + "\n\n" + user_prompt

    # Initialize active sequences with search and URL fetch counters
    active_sequences = [{
        'item': query,
        'prompt': full_prompt,
        'output': '',
        'finished': False,
        'history': [],
        'pending_operations': [],  # Queue of operations to execute
        'executed_search_queries': set(),
        'executed_url_fetches': set(),
        'search_count': 0  # Search counter
    }]

    start_time = time.time()
    turn = 0
    url_fetch = None
    while True:
        # Separate sequences with pending operations and those needing generation
        sequences_with_pending_ops = [seq for seq in active_sequences if
                                      not seq['finished'] and seq['pending_operations']]
        sequences_needing_generation = [seq for seq in active_sequences if
                                        not seq['finished'] and not seq['pending_operations']]
        # First, handle pending operations
        if sequences_with_pending_ops:
            print(f"{len(sequences_with_pending_ops)} sequences have pending operations. Executing...")
            for seq in sequences_with_pending_ops:
                # Execute the next pending operation
                operation = seq['pending_operations'].pop(0)  # FIFO
                op_type = operation['type']
                content = operation['content']

                if op_type == 'search':
                    query = content
                    try:
                        # Execute search and cache results
                        results = exa_web_search(query, url)
                        print(f"Executed and cached search for query: {query}")
                    except Exception as e:
                        print(f"Error during search query '{query}': {e}")
                        results = {}
                    relevant_info = extract_relevant_info(results)[:top_k]
                    search_result_str = json.dumps(relevant_info, ensure_ascii=False, indent=2)
                    # Append search results to the prompt
                    append_text = f"\n{BEGIN_SEARCH_RESULT}\n{search_result_str}\n{END_SEARCH_RESULT}\n"
                    seq['prompt'] += append_text
                    seq['output'] += append_text
                    # Update history
                    seq['history'].append(append_text)
                    # Increment search count
                    seq['search_count'] += 1

                elif op_type == 'fetch_url':
                    urls = content
                    # Calculate remaining URL fetches
                    remaining_fetches = MAX_URL_FETCH - len(seq['executed_url_fetches'])
                    if remaining_fetches <= 0:
                        # Reached URL fetch limit, add limit message and mark sequence as finished
                        limit_message = f"\n{BEGIN_FULL_PAGE}\nThe maximum number of URL fetches has been reached. You are not allowed to fetch more URLs.\n{END_FULL_PAGE}\n"
                        seq['prompt'] += limit_message
                        seq['output'] += limit_message
                        seq['history'].append(limit_message)
                        print("Reached URL fetch limit. Sequence marked as finished.")
                        continue

                    # Split and clean URLs
                    urls_to_fetch = [u.strip() for u in urls.split(",")]
                    # Filter already fetched URLs
                    urls_to_fetch = [u for u in urls_to_fetch if u not in seq['executed_url_fetches']]
                    # Limit the number of URLs to fetch
                    urls_to_fetch = urls_to_fetch[:remaining_fetches]

                    if not urls_to_fetch:
                        print("All requested URLs have been fetched or exceeded the limit.")
                        continue

                    # Batch fetch page content, considering cache
                    urls_to_fetch_filtered = [u for u in urls_to_fetch if u not in url_cache]
                    cached_urls = [u for u in urls_to_fetch if u in url_cache]

                    fetched_contents = []

                    # Use cached URL content
                    for url in cached_urls:
                        content = url_cache[url]
                        print(f"Using cached URL content for URL: {url}")
                        fetched_contents.append((url, content))

                    # Batch fetch uncached URLs
                    if urls_to_fetch_filtered:
                        try:
                            # Batch pass uncached URLs
                            contents = fetch_page_content(urls_to_fetch_filtered)
                            for url, content in contents.items():
                                url_cache[url] = content
                                print(f"Fetched and cached URL content for URL: {url}")
                                fetched_contents.append((url, content))
                        except Exception as e:
                            for url in urls_to_fetch_filtered:
                                content = f"Error fetching URL: {e}"
                                url_cache[url] = content
                                fetched_contents.append((url, content))
                                print(f"Error fetching URL '{url}': {e}")

                    # Update fetched URLs
                    for url, _ in fetched_contents:
                        seq['executed_url_fetches'].add(url)

                    # Construct full page content string
                    fetched_pages = dict(fetched_contents)
                    full_page_str = json.dumps(fetched_pages, ensure_ascii=False, indent=2)
                    # Append full page content to the prompt
                    append_text = f"\n{BEGIN_FULL_PAGE}\n{full_page_str}\n{END_FULL_PAGE}\n"
                    seq['prompt'] += append_text
                    seq['output'] += append_text
                    # Update history
                    seq['history'].append(append_text)

                    print(f"Fetched and cached {len(fetched_contents)} URLs.")

                    # Check if URL fetch limit is reached
                    if len(seq['executed_url_fetches']) >= MAX_URL_FETCH:
                        limit_message = f"\n{BEGIN_FULL_PAGE}\nThe maximum number of URL fetches has been reached. You are not allowed to fetch more URLs.\n{END_FULL_PAGE}\n"
                        seq['prompt'] += limit_message
                        seq['output'] += limit_message
                        seq['history'].append(limit_message)
                        print("Reached URL fetch limit. Sequence marked as finished.")

        # Continue to the next iteration if there are pending operations
        if sequences_with_pending_ops:
            continue  # Process operations first

        # Handle sequences needing generation
        if sequences_needing_generation:
            turn += 1
            print(f"Turn {turn}: {len(sequences_needing_generation)} sequences need generation. Generating with LLM...")
            outputs = run_generation(sequences_needing_generation, model_url)
            print("*"*100)
            print(outputs[0])
            print("*" * 100)
            print("Generation complete. Processing outputs...")

            # Process each generated output
            for seq, out in zip(sequences_needing_generation, outputs):
                text = out
                seq['history'].append(text)
                # Append generated text to prompt and output
                seq['prompt'] += text
                seq['output'] += text

                # Check if the generated content contains search queries or URL fetch requests
                search_query = extract_between(text, BEGIN_SEARCH_QUERY, END_SEARCH_QUERY)

                # 如果turn==0，那么将不将url提取出来
                if turn != 0:
                    url_fetch = extract_between(text, BEGIN_URL, END_URL)

                if search_query:
                    # Check if search limit is not exceeded
                    if seq['search_count'] < MAX_SEARCH_LIMIT:
                        # Check if this search query has not been executed
                        if search_query not in seq['executed_search_queries']:
                            # Add search operation to pending queue
                            seq['pending_operations'].append({'type': 'search', 'content': search_query})
                            seq['executed_search_queries'].add(search_query)
                            print(f"Added pending search operation for query: {search_query}")
                        else:
                            print(f"Search query already executed: {search_query}")
                    else:
                        # Add limit message if search limit is exceeded
                        limit_message = f"\n{BEGIN_SEARCH_RESULT}\nThe maximum search limit is exceeded. You are not allowed to search.\n{END_SEARCH_RESULT}\n"
                        seq['prompt'] += limit_message
                        seq['output'] += limit_message
                        seq['history'].append(limit_message)
                        print(f"Search limit exceeded for query: {search_query}")

                if url_fetch:
                    # Check if URL fetch limit is not exceeded
                    if len(seq['executed_url_fetches']) < MAX_URL_FETCH:
                        # Split and check if URLs have already been fetched
                        urls = [u.strip() for u in url_fetch.split(",")]
                        urls_to_fetch = [u for u in urls if u not in seq['executed_url_fetches']]
                        if urls_to_fetch:
                            # Add URL fetch operation to pending queue
                            seq['pending_operations'].append({'type': 'fetch_url', 'content': ', '.join(urls_to_fetch)})
                            print(f"Added pending URL fetch operation for URLs: {urls_to_fetch}")
                        else:
                            print(f"All requested URLs have been fetched or exceeded the limit: {urls}")
                    else:
                        # Add limit message if URL fetch limit is exceeded
                        limit_message = f"\n{BEGIN_FULL_PAGE}\nThe maximum number of URL fetches has been reached. You are not allowed to fetch more URLs.\n{END_FULL_PAGE}\n"
                        seq['prompt'] += limit_message
                        seq['output'] += limit_message
                        seq['history'].append(limit_message)
                        print("URL fetch limit exceeded.")

                # If no new operations are added, mark sequence as finished
                if not search_query and not url_fetch:
                    seq['finished'] = True
                    print("Sequence marked as finished.")

        # Check if all sequences are finished
        unfinished = [seq for seq in active_sequences if not seq['finished']]
        if not unfinished:
            break
        else:
            if turn >= MAX_TURN:
                print(f"Exceeded maximum number of turns ({MAX_TURN}). Stopping.")
                break
            # Optionally, implement a delay or other logic to prevent infinite loops
            pass

        print("=" * 100)
        print("=" * 100)

    total_time = time.time() - start_time
    output_list = [seq['output'] for seq in active_sequences]

    if os.path.exists(url_cache_path):
        with open(url_cache_path, 'r', encoding='utf-8') as f:
            url_cache_new = json.load(f)
    else:
        url_cache_new = {}

    # print(sequences_needing_generation[0]["prompt"])
    url_cache.update(url_cache_new)

    save_caches(url_cache_path, url_cache)

    return sequences_needing_generation[0]["output"]


if __name__ == '__main__':
    model_url = "http://192.168.16.4:8000/v1"
    message = "刘翔获得了多少个冠军？"
    message = "《哪吒之魔童闹海》目前的票房是多少？位于全球票房第几？"
    message = "小米su7 ultra定价多少钱，和su7相比有什么变化？"
    response = rag_agent_main(message, model_url)
    print(response)
