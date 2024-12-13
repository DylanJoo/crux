from openai import OpenAI

class API:

    def __init__(self, args, url=None):
        self.args = args
        self.client = OpenAI(
            base_url=f"http://0.0.0.0:{args.port}/v1",
            api_key="EMPTY"
        )
        self.completion = None

    @property
    def prompt_len(self):
        return self.completion.usage.prompt_tokens

    def generate(self, x, **kwargs):

        self.completion = self.client.completions.create(
            model=self.args.model,
            prompt=x,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            max_tokens = kwargs.pop('max_tokens', 256),
        )
        output = self.completion.choices[0].text
        return output

# Using CURL
# body = """{ 
#     "model": "<model>",
#     "temperature": <temperature>,
#     "top_p": <top_p>,
#     "max_tokens": <max_tokens>,
#     "min_tokens": <min_tokens>,
#     "prompt": "<prompt>"
# }"""
# body = body.replace("<model>", self.args.model)
# body = body.replace("<temperature>", str(self.args.temperature))
# body = body.replace("<top_p>", str(self.args.top_p))
# body = body.replace("<max_tokens>", str(max_tokens))
# body = body.replace("<min_tokens>", str(min_tokens))
# # body = body.replace("<prompt>", x)
# print(body)
