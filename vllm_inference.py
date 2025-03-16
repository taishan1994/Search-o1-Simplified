# coding=utf-8
import os

from tqdm import tqdm

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
from transformers import AutoTokenizer
import requests
import json
import time
import concurrent.futures


def chat_with_model(message, model, model_url):
    """
    Simple function to chat with a reasoning-enabled model.

    Args:
        message (str): The message to send to the model
        model_url (str): The base URL of the model server
    """

    # Prepare the chat message
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": message}],
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": 32768,
    }

    # Make the API call
    response = requests.post(
        f"{model_url}/chat/completions",
        headers={"Authorization": "Bearer EMPTY"},
        json=payload
    )

    # print(response.content)

    if response.status_code == 200:
        result = response.json()
        reasoning = result["choices"][0]["message"].get("reasoning_content", "")
        content = result["choices"][0]["message"].get("content", "")

        # print(reasoning)
        # print(content)

        if reasoning:
            print(f"Reasoning: {reasoning}")
        print(f"Content: {content}")
        return content
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None


def chat_without_model(message, model_url):
    """
    Simple function to chat with a reasoning-enabled model.

    Args:
        message (str): The message to send to the model
        model_url (str): The base URL of the model server
    """
    models_response = requests.get(
        f"{model_url}/models",
        headers={"Authorization": "Bearer EMPTY"}
    )

    # print(models_response.content)

    model = models_response.json()["data"][0]["id"]
    # print(model)

    # Prepare the chat message
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": message}],
        "temperature": 1,
        "top_p": 0.95,
        "max_tokens": 10000,
    }

    # Make the API call
    response = requests.post(
        f"{model_url}/chat/completions",
        headers={"Authorization": "Bearer EMPTY"},
        json=payload
    )

    # print(response.content)

    if response.status_code == 200:
        result = response.json()
        reasoning = result["choices"][0]["message"].get("reasoning_content", "")
        content = result["choices"][0]["message"].get("content", "")

        # print(reasoning)
        # print(content)

        # if reasoning:
        #     print(f"Reasoning: {reasoning}")
        # print(f"Content: {content}")
        return content
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

def chat_without_model2(args):
    """
    Simple function to chat with a reasoning-enabled model.

    Args:
        message (str): The message to send to the model
        model_url (str): The base URL of the model server
    """
    i, message, model_url = args
    models_response = requests.get(
        f"{model_url}/models",
        headers={"Authorization": "Bearer EMPTY"}
    )

    # print(models_response.content)

    model = models_response.json()["data"][0]["id"]
    # print(model)

    # Prepare the chat message
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": message}],
        "temperature": 1,
        "top_p": 0.95,
        "max_tokens": 10000,
    }

    # Make the API call
    response = requests.post(
        f"{model_url}/chat/completions",
        headers={"Authorization": "Bearer EMPTY"},
        json=payload
    )

    # print(response.content)

    if response.status_code == 200:
        result = response.json()
        reasoning = result["choices"][0]["message"].get("reasoning_content", "")
        content = result["choices"][0]["message"].get("content", "")

        # print(reasoning)
        # print(content)

        # if reasoning:
        #     print(f"Reasoning: {reasoning}")
        # print(f"Content: {content}")
        return {"idx":i, "content":content}
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None


if __name__ == '__main__':
    model_url = "http://xxxx:8000/v1"
    message = "《哪吒之魔童闹海的目前票房是多少？》"

    print(message)
    print(chat_without_model(message, model_url))
