import requests
import json
import os

ShangLiang_cofiguration = {
        'request_url': 'https://lm_experience.sensetime.com/v1/nlp/chat/completions',
        'model_config':
            {"temperature": 0.5,
             "top_p": 0.7,
             "max_new_tokens": 2048,
             "repetition_penalty": 1,
             "stream": False,
             "user": "test"}
        }

def get_shangliang_res(prompt, key="b662fb5453e844499ffdc53547ae7951"):
    url = ShangLiang_cofiguration['request_url']
    data = {
        "messages": [{"role": "user", "content": prompt}],
    }
    data.update(ShangLiang_cofiguration['model_config'])
    print(data)
    headers = {
        'Content-Type': 'application/json',
        'Authorization': key
    }
    response = requests.post(url, headers=headers, json=data)
    res = json.loads(response.text)['data']['choices'][0]['message']
    return res


if __name__ == '__main__':
    prompt = input("请输入您要说的话：")
    res = get_shangliang_res(prompt)
    print(res)