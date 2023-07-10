import os
import openai

# openai.api_key = "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"
# models = openai.Model.list()['data']
# model_ids = []
# for model in models:
#     model_ids.append(model['id'])
# print(model_ids)
# model_id = "gpt-3.5-turbo"
# openai.Model.retrieve(model_id)

import requests
import json



def make_request(content):
    openai.api_key = "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"
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content}
        ]
    )
    return completion.choices[0].message['content']

if __name__=='__main__':
    while True:
        contents =input()
        print(make_request(contents))
