import requests

# ----------------------- Custom Headers -----------------------

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/58.0.3029.110 Safari/537.36',
    'Referer': 'https://www.google.com/',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1'
}

# Initialize session
session = requests.Session()
session.headers.update(headers)


def do_request():
    url = "https://api.exa.ai/search"

    payload = {
        "query": "《哪吒之魔童闹海》的目前票房是多少？",
        "useAutoprompt": False,
        "type": "auto",
        "numResults": 10,
        "text": True,
    }
    headers = {
        "x-api-key": "xxx",
        "Content-Type": "application/json"
    }

    response = requests.request("POST", url, json=payload, headers=headers)

    print(response.text)

from exa_py import Exa

exa = Exa(api_key = "xxx")

result = exa.search_and_contents(
  "《哪吒之魔童闹海》的目前票房是多少？",

  text = True
)
print(result)

# if __name__ == '__main__':
#     do_request()
