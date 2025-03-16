from vllm_inference import chat_without_model


def direct_main(message, model_url):
    response = chat_without_model(message, model_url)
    return response


if __name__ == '__main__':
    model_url = "http://192.168.16.4:8000/v1"
    message = "刘翔获得了多少个冠军？"
    message = "《哪吒之魔童闹海》目前的票房是多少？位于全球票房第几？"
    response = direct_main(message, model_url)
    print(response)
