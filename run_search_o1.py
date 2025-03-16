import os
import re
import time
import json
import concurrent.futures
from pprint import pprint
from typing import Optional, List, Dict
from datetime import datetime
from tqdm import tqdm

from exa_search_main import (
    extract_snippet_with_context,
    exa_web_search,
    extract_relevant_info,
    extract_snippet_with_context,
    fetch_page_content,
)
from vllm_inference import chat_without_model, chat_without_model2

BEGIN_SEARCH_QUERY = "<|begin_search_query|>"
END_SEARCH_QUERY = "<|end_search_query|>"
BEGIN_SEARCH_RESULT = "<|begin_search_result|>"
END_SEARCH_RESULT = "<|end_search_result|>"
BEGIN_URL = "<|begin_url|>"
END_URL = "<|end_url|>"
BEGIN_FULL_PAGE = "<|begin_full_page|>"
END_FULL_PAGE = "<|end_full_page|>"
MAX_SEARCH_LIMIT = 8
MAX_URL_FETCH = 100
MAX_TURN = 10
max_tokens = 32768
top_k = 10
max_doc_len = 3000
current_time = datetime.now()


def run_generation(sequences_needing_generation, model_url):
    # print(sequences_needing_generation[0]["prompt"])
    result = chat_without_model(sequences_needing_generation[0]["prompt"], model_url)
    return [result]


# Function to extract text between two tags
def extract_between(text: str, start_tag: str, end_tag: str):
    pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
    matches = re.findall(pattern, text, flags=re.DOTALL)
    # print(matches)
    # 这里需要进行修改，不能只返回最后一个
    if matches:
        # return matches[-1].strip()
        return [i.strip() for i in matches]
    return None


def save_caches(url_cache_path, url_cache):
    with open(url_cache_path, 'w', encoding='utf-8') as f:
        json.dump(url_cache, f, ensure_ascii=False, indent=2)


def get_webpage_to_reasonchain_instruction(prev_reasoning, search_query, document):
    webpage_to_resonchain_instruction = f"""**Task Instruction:**

    You are tasked with reading and analyzing web pages based on the following inputs: **Previous Reasoning Steps**, **Current Search Query**, and **Searched Web Pages**. Your objective is to extract relevant and helpful information for **Current Search Query** from the **Searched Web Pages** and seamlessly integrate this information into the **Previous Reasoning Steps** to continue reasoning for the original question.

    **Guidelines:**

    1. **Analyze the Searched Web Pages:**
    - Carefully review the content of each searched web page.
    - Identify factual information that is relevant to the **Current Search Query** and can aid in the reasoning process for the original question.

    2. **Extract Relevant Information:**
    - Select the information from the Searched Web Pages that directly contributes to advancing the **Previous Reasoning Steps**.
    - Ensure that the extracted information is accurate and relevant.

    3. **Output Format:**
    - **If the web pages provide helpful information for current search query:** Present the information beginning with `**Final Information**` as shown below.
    **Final Information**

    [Helpful information]

    - **If the web pages do not provide any helpful information for current search query:** Output the following text.

    **Final Information**

    No helpful information found.

    **Inputs:**
    - **Previous Reasoning Steps:**  
    {prev_reasoning}

    - **Current Search Query:**  
    {search_query}

    - **Searched Web Pages:**  
    {document}

    Now you should analyze each web page and find helpful information based on the current search query "{search_query}" and previous reasoning steps.
    """
    return webpage_to_resonchain_instruction


def extract_answer(output, mode='gen'):
    extracted_text = ''
    if mode == 'codegen':
        # Extract the code between ```python and ```
        pattern = r'```python\s*(.*?)\s*```'
        matches = re.findall(pattern, output, re.DOTALL | re.IGNORECASE)
        if matches:
            extracted_text = matches[-1].strip()  # Take the last match
    elif mode == 'infogen':
        # Extract content after **Final Information** or **Modified Reasoning Steps**
        pattern_info = "**Final Information**"
        pattern_step = "**Modified Reasoning Steps**"
        if pattern_info in output:
            extracted_text = output.split(pattern_info)[-1].replace("\n", "").strip("```").strip()
        elif pattern_step in output:
            extracted_text = output.split(pattern_step)[-1].strip("```").strip()
        else:
            extracted_text = "No helpful information found."
    else:
        # Existing extraction logic for 'gen' and 'choose' modes
        pattern = r'\\boxed\{(.*)\}'
        matches = re.findall(pattern, output)
        if matches:
            extracted_text = matches[-1]  # Take the last match
            if mode in ['choose', 'qa']:
                # Handle 'choose' mode
                inner_pattern = r'\\text\{(.*)\}'
                inner_matches = re.findall(inner_pattern, extracted_text)
                if inner_matches:
                    extracted_text = inner_matches[-1]  # Take the last match
                extracted_text = extracted_text.strip("()")
    return extracted_text


def generate_webpage_to_reasonchain_batch(
        original_questions: List[str],
        prev_reasonings: List[str],
        search_queries: List[str],
        documents: List[str],
        batch_output_records: List[Dict],  # New parameter to collect outputs
        model_url: str = None,
):
    # 这里进行修改，每一个url都调用模型进行判断
    user_prompts = []
    for r, sq, doc in zip(prev_reasonings, search_queries, documents):
        for d in tqdm(doc, total=len(doc)):
            user_prompts.append((sq, get_webpage_to_reasonchain_instruction(r, sq, d)))

    prompts = [{"role": "user", "content": up, "query": sq} for sq, up in user_prompts]

    raw_outputs = []
    num_workers = 10
    with concurrent.futures.ProcessPoolExecutor(num_workers) as executor:
        # 提交任务并获得 future 对象
        futures = [executor.submit(chat_without_model2, (i, up[1][:25000], model_url)) for i,up in enumerate(user_prompts)]
        # for future in tqdm(concurrent.futures.as_completed(futures)):
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(user_prompts)):
            result = future.result()
            # print(result["result"]["labels"])
            raw_outputs.append(result)
    raw_outputs = sorted(raw_outputs, key=lambda x:x["idx"])
    # print(raw_outputs)
    raw_outputs = [i["content"] for i in raw_outputs]
    # for up in user_prompts:
    #     output = chat_without_model(up[:25000], model_url)
    #     raw_outputs.append(output)

    extracted_infos = [extract_answer(raw, mode='infogen') for raw in raw_outputs]

    for i, (p, r, e) in enumerate(zip(prompts, raw_outputs, extracted_infos)):
        batch_output_records.append({
            "query": p["query"],
            'prompt': p,
            'raw_output': r,
            'extracted_info': e
        })

    return batch_output_records


def run_search_o1_main(query, model_url):
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
        "You are a reasoning assistant with the ability to perform web searches to help "
        "you answer the user's question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"Who got the first Nobel Prize in Physics?\"\n"
        "Assistant thinking steps:\n"
        "- I need to find out who was awarded the first Nobel Prize in Physics.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>first Nobel Prize in Physics winner<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant continues reasoning with the new information...\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- When done searching, continue your reasoning.\n\n"
    )

    # 'Provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'
    user_prompt = (
        'Please answer the following question. You should think step by step to solve it.\n\n'
        f'Current time: {current_time}\n\n'
        f'Question:\n{query}\n\n'
    )

    full_prompt = instruction + "\n\n" + user_prompt

    # ---------------------- Preparation of Input Prompts ----------------------
    # Initialize active sequences
    active_sequences = [{
        'item': query,
        'prompt': full_prompt,
        'output': '',
        'finished': False,
        'history': [],
        'search_count': 0,
        'executed_search_queries': set(),
    }]

    # ---------------------- Initialize Collection Structure ----------------------
    # Initialize a list to collect batch outputs
    batch_output_records = []

    start_time = time.time()
    turn = 0

    # Main loop until all sequences are finished or maximum turns reached
    while True:
        # Identify sequences that need generation
        # 找到还需要进行生成的seq
        sequences_needing_generation = [seq for seq in active_sequences if not seq['finished']]

        if sequences_needing_generation:
            turn += 1
            print(f'\n-------------- Turn {turn} --------------')
            print(f"We have {len(sequences_needing_generation)} sequences needing generation...")
            # 第一轮，模型根据用户的query，拆分为多个需要的查询
            outputs = run_generation(sequences_needing_generation, model_url)
            print("Generation completed, processing outputs...")

            # Initialize batch variables
            batch_relevant_info = []
            batch_original_questions = []
            batch_prev_reasonings = []
            batch_search_queries = []
            batch_documents = []
            batch_sequences = []

            # Collect URLs to fetch across all sequences
            all_urls_to_fetch = set()
            url_snippets = {}
            url_sequence_map = {}  # Map URL to list of sequences needing it

            # Process each sequence and collect URLs
            for seq, out in zip(sequences_needing_generation, outputs):
                text = outputs[0]
                seq['history'].append(text)
                # Append generated text to prompt and output
                if turn == 1:
                    # 第一轮的时候需要后面的推理结果，因为可能会产生幻觉
                    if "<|end_search_query|>" not in text:
                        raise Exception("模型没有正确输出<|end_search_query|>，请检查！！")
                    tmp_text = text.split("\n\n")
                    text = tmp_text[0]
                    for t in tmp_text[1:]:
                        if "<|end_search_query|>" in t:
                            text += t

                seq['prompt'] += text
                seq['output'] += text

                # Extract search query
                search_query = extract_between(text, BEGIN_SEARCH_QUERY, END_SEARCH_QUERY)

                # If a search query is present and needs to be executed
                if search_query:
                    for i, sq in enumerate(search_query):
                        if seq['search_count'] < MAX_SEARCH_LIMIT and sq not in seq['executed_search_queries']:
                            # Execute search, use cache if available
                            try:
                                results = exa_web_search(sq, url)
                                print(f"Executed and cached search for query: \"{sq}\"")
                            except Exception as e:
                                print(f"Error during search query '{sq}': {e}")
                                results = {}

                            # Extract relevant information from Bing search results
                            relevant_info = extract_relevant_info(results)[:top_k]
                            seq['relevant_info'] = relevant_info

                            # Extract URLs and snippets
                            urls_to_fetch = [it['url'] for it in relevant_info]
                            snippets = {info['url']: info['snippet'] for info in relevant_info if 'snippet' in info}

                            # Filter URLs that are not cached
                            urls_to_fetch_filtered = [u for u in urls_to_fetch if u not in url_cache]
                            cached_urls = [u for u in urls_to_fetch if u in url_cache]

                            # Store info for all_urls_to_fetch and url_snippets
                            for url in urls_to_fetch_filtered:
                                all_urls_to_fetch.add(url)
                                url_snippets[url] = snippets.get(url, "")

                            all_reasoning_steps = seq['output']

                            truncated_prev_reasoning = ""
                            truncated_prev_reasoning += f"{all_reasoning_steps}\n\n"

                            # Collect parameters for batch processing
                            batch_relevant_info.append(relevant_info)
                            batch_original_questions.append(seq['item'])
                            batch_prev_reasonings.append(truncated_prev_reasoning)
                            batch_search_queries.append(sq)
                            batch_sequences.append(seq)

                            # Update search count and executed queries
                            seq['search_count'] += 1
                            seq['executed_search_queries'].add(sq)

                        elif seq['search_count'] >= MAX_SEARCH_LIMIT:
                            limit_message = f"\n{BEGIN_SEARCH_RESULT}\nThe maximum search limit is exceeded. You are not allowed to search.\n{END_SEARCH_RESULT}\n"
                            seq['prompt'] += limit_message
                            seq['output'] += limit_message
                            seq['history'].append(limit_message)
                            print(f"Search limit reached for query: \"{sq}\"")

                        elif sq in seq['executed_search_queries']:
                            limit_message = f"\n{BEGIN_SEARCH_RESULT}\nYou have searched this query. Please refer to previous results.\n{END_SEARCH_RESULT}\n"
                            seq['prompt'] += limit_message
                            seq['output'] += limit_message
                            seq['history'].append(limit_message)
                            print(f"Repeated search for query: \"{sq}\"")
                else:
                    # If no search query needs to be executed, mark the sequence as finished
                    seq['finished'] = True
                    print("Sequence marked as complete.")

            # Batch fetch all URLs at once to optimize speed
            if all_urls_to_fetch:
                print(f"Fetching {len(all_urls_to_fetch)} URLs...")
                try:
                    fetched_contents = fetch_page_content(
                        list(all_urls_to_fetch),
                        # snippets=url_snippets  # Do not pass snippets when updating url_cache directly
                    )
                    print(f"Fetched {len(fetched_contents)} URLs successfully.")
                except Exception as e:
                    print(f"Error during batch URL fetching: {e}")
                    fetched_contents = {url: f"Error fetching URL: {e}" for url in all_urls_to_fetch}
                # Update cache with fetched contents
                for url, content in fetched_contents.items():
                    url_cache[url] = content

            # After fetching, prepare formatted documents for batch processing
            for relevant_info in batch_relevant_info:
                tmp = []
                for i, doc_info in enumerate(relevant_info):
                    url = doc_info['url']
                    raw_context = url_cache.get(url, "")
                    doc_info['snippet'] = doc_info['snippet'].replace('<b>', '').replace('</b>', '')
                    success, filtered_context = extract_snippet_with_context(raw_context, doc_info['snippet'],
                                                                             context_chars=max_doc_len)
                    if success:
                        context = filtered_context
                    else:
                        context = raw_context[:max_doc_len * 2]

                    doc_info['context'] = context
                    tmp.append(json.dumps(doc_info, ensure_ascii=False, indent=2) + "\n")

                batch_documents.append(tmp)

            # After fetching, prepare for batch processing if there are any
            if batch_sequences:
                print(
                    f"Batch processing {len(batch_sequences)} sequences with generate_webpage_to_reasonchain_batch...")
                webpage_analyses = generate_webpage_to_reasonchain_batch(
                    original_questions=batch_original_questions,
                    prev_reasonings=batch_prev_reasonings,
                    search_queries=batch_search_queries,
                    documents=batch_documents,
                    batch_output_records=batch_output_records,  # Pass the collection list
                    model_url=model_url,
                )
                print("Batch generation completed, assigning outputs to sequences...")

                for seq in batch_sequences:
                    relevant_info = seq["relevant_info"]
                    append_text_dict = {}
                    for rel, analysis in zip(relevant_info, webpage_analyses):
                        query = analysis["query"]
                        extract_info = analysis["extracted_info"]
                        if query not in append_text_dict:
                            append_text_dict[query] = ""
                        date = rel["date"]
                        if date == "":
                            date = "无法获取page发布时间\t"
                        else:
                            date = date + "\t"
                        append_text_dict[
                            query] += f"\n\n{BEGIN_SEARCH_RESULT}{date}{extract_info}{END_SEARCH_RESULT}\n\n"

                    for k, v in append_text_dict.items():
                        append_text = f"\n\n{BEGIN_SEARCH_QUERY}{k}{END_SEARCH_QUERY}\n\n" + v + "\n\n"
                        seq['prompt'] += append_text
                        seq['output'] += append_text
                        seq['history'].append(append_text)

                print("====模型总结结果====")
                print(batch_sequences[0]['prompt'])
                print("====模型总结结果====")
                active_sequences = batch_sequences

        # Check if all sequences are finished
        unfinished = [seq for seq in active_sequences if not seq['finished']]
        if not unfinished:
            break
        else:
            if turn >= MAX_TURN:
                print(f"Maximum number of turns ({MAX_TURN}) reached, stopping.")
                break

    total_time = time.time() - start_time

    # ---------------------- Save Batch Output Records to JSON File ----------------------
    # Define output JSON file path
    t = time.localtime()

    if os.path.exists(url_cache_path):
        with open(url_cache_path, 'r', encoding='utf-8') as f:
            url_cache_new = json.load(f)
    else:
        url_cache_new = {}

    # Prepare output list for evaluation
    output_list = [seq['output'] for seq in active_sequences]
    print("=" * 100)
    print(output_list[0])

    url_cache.update(url_cache_new)

    save_caches(url_cache_path, url_cache)

    print("Process completed.")


if __name__ == "__main__":
    model_url = "http://192.168.16.4:8000/v1"
    model_url = "http://192.168.16.6:9001/v1"
    message = "刘翔获得了多少个冠军？"
    message = "《哪吒之魔童闹海》目前的票房是多少？位于全球票房第几？"
    # message = "梳理一下《哪吒之魔童闹海》的票房的发展趋势"
    message = "manus是什么？"
    message = "W8A8量化有什么方法吗"
    # message = "姚明和奥尼尔哪一个更厉害？"
    response = run_search_o1_main(message, model_url)
    print(response)
