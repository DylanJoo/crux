from fidt5 import FiDT5
from transformers import AutoTokenizer

MODEL_PATH='google/flan-t5-large'
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
MODEL_PATH='/ivi/ilps/personal/dju/checkpoints/ctxcomp-flan-t5-arge-inverted-mds/checkpoint-5000'
model = FiDT5.from_pretrained(MODEL_PATH)

question_template = "Summarize each documents based on the topic. Write the summary with the document identifier (a number with square brackets). Only provide the summary for relevant documents and ignore the empty document. If the document is not relevant to the topic, write `irrelevant` instead. Topic: {}"
doc_template = "Document [{}]: {}\n"
summary_template = "[{}]: {}\n"

B = 2
N = 3 + 1
inputs = tokenizer([
    question_template.format("what is xxx512?"), 
    'Document [1]: xxe512 is a dog.', 
    'Document [2]: x3x512 is a cat.', 
    'Document [3]: xxx512 is a car.', 
    question_template.format('what is banana?'),
    'Document [1]: bana is a dog', 
    'Document [2]: apple is a fruit.', 
    'Document [3]: hanana is a car', 
], padding=True, truncation=True, return_tensors='pt')

input_ids = inputs['input_ids'].view(B, N, -1)
attention_mask = inputs['attention_mask'].view(B, -1)


output = model.generate(
    input_ids=input_ids, 
    attention_mask=attention_mask,
    max_new_tokens=512
)
print(output)
print(tokenizer.decode(output[0]))
print(tokenizer.decode(output[1]))

# ## original fidt5
# inputs = tokenizer([
#     'question: what is xxx512? context: xxe512 is a dog.', 
#     'question: what is xxx512? context: x3x512 is a cat.', 
#     'question: what is xxx512? context: xxx512 is a car.', 
#     'question: what is banana? context: banana is not a dog', 
#     'question: what is banana? context: banana is not a fruit.', 
#     'question: what is banana? context: banana is a car', 
# ], padding=True, truncation=True, return_tensors='pt')
#
# input_ids = inputs['input_ids'].view(2, 3, -1)
# attention_mask = inputs['attention_mask'].view(2, -1)
#
#
# output = model.generate(input_ids=input_ids, attention_mask=attention_mask)
# print(output)
# print(tokenizer.decode(output[0]))
# print(tokenizer.decode(output[1]))
