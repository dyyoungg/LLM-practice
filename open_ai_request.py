import requests
import json
from tool import Tokennizer

class OpenAI_Request(object):

    def __init__(self,key, model_name, request_address, generate_config=None):
        super().__init__()
        self.headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        self.model__name = model_name
        self.request_address = request_address
        self.generate_config = generate_config

    def post_request(self, message):

        data = {
            "model": self.model__name,
            "messages":  message
        }

        # add generate parameter of api
        if self.generate_config:
            for k,v in self.generate_config.param_dict.__dict__.items():
                data[k] = v

        data = json.dumps(data)

        response = requests.post(self.request_address, headers=self.headers, data=data)

        return response

class ContextHandler(object):
    def __init__(self,max_context=3200):
        super().__init__()
        self.context = []
        self.role_lengths = []
        self.max_context = max_context

    def append_cur_to_context(self,data,complete__length, tag=0):

        if tag == 0:
            role = "user"
        elif tag == 1:
            role = "assistant"
        else:
            role = "system"

        role_data = {"role": role, "content": data}
        self.context.append(role_data)
        self.role_lengths.append(complete__length)

    def cut_context(self,cur_total_length,tokenizer):
        if cur_total_length > self.max_context:
            self.context = self.context[self.max_context * 0.5:]
            self.role_lengths = self.role_lengths[ self.max_context * 0.5]

    def clear(self):
        self.context.clear()



if __name__ == '__main__':
    keys = "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"
    model_name = "gpt-3.5-turbo"
    request_address = "https://api.openai.com/v1/chat/completions"
    requestor = OpenAI_Request(keys, model_name, request_address)
    tokenizer = Tokennizer(model_name=model_name)
    contextHandler = ContextHandler()
    context_max = 3200
    while True:
        input_s = input('\nuser: ')
        if input_s == "clear":
            contextHandler.clear()
            print('new sessions')
            continue
        else:
            token_length = tokenizer.num_tokens_from_string(input_s)
            contextHandler.append_cur_to_context(input_s, token_length)
        res = requestor.post_request(contextHandler.context)

        if res.status_code == 200:

            response = res.json()['choices'][0]['message']['content']
            # print(response)
            # print(contextHandler.context, contextHandler.role_lengths)
            # cut \n for show
            response = response.lstrip("\n")

            completion_length = res.json()['usage']['completion_tokens']
            total_length = res.json()['usage']['total_tokens']
            # print('total tokens:', total_length)
            print(f"\nresponse : {response}")


            contextHandler.append_cur_to_context(response, completion_length, tag=1)
            if total_length > context_max:
                contextHandler.cut_context(total_length, tokenizer)
        else:
            status_code = res.status_code
            reason = res.reason
            des = res.text
            raise print(f'visit error :\n status code: {status_code}\n reason: {reason}\n err description: {des}\n '
                        f'please check whether your account  can access OpenAI API normally')
