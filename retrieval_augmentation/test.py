from transformers import AutoTokenizer
from datasets import load_dataset
from utils import update_tokenizer

ds = load_dataset('json', data_files='/home/dju/rag-rerank/data/inverted-mds/test.jsonl', keep_in_memory=True)['train']
batched = [ds[0], ds[1], ds[2] ]
batched = [ds[0]]

tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-large')
tokenizer = update_tokenizer(tokenizer)

from data import Standard
d1 = Standard(
    tokenizer=tokenizer,
    max_src_length=1023,
    max_tgt_length=512,
    n_contexts=None
)
a = d1(batched)

# inputs
# b = a.input_ids[:, :, :100].contiguous()
# B, N, L = b.shape
#
# o = tokenizer.batch_decode(b.view(B, N*L), skip_special_tokens=False)
# print(o[0])
# print('label')
# a.labels[a.labels == -100] = 0
# print(tokenizer.decode(a.labels[0], skip_special_tokens=True))
# print(tokenizer.decode(a.labels[1], skip_special_tokens=True))

from models import FiDT5
# MODEL_PATH='/ivi/ilps/personal/dju/checkpoints/ctxcomp-flan-t5-arge-inverted-mds-std/checkpoint-5000'
# model = FiDT5.from_pretrained(MODEL_PATH)
#
# # standard
# output = model.generate(
#     input_ids=a.input_ids, 
#     attention_mask=a.attention_mask,
#     do_sample=True,
#     top_p=0.1,
#     temperature=0.2,
#     min_new_tokens=32,
#     max_new_tokens=512,
# )
#
# print('src', tokenizer.decode(a.input_ids[:, :, :50].contiguous().view(-1), skip_special_tokens=True))
# print('pred', tokenizer.decode(output[0], skip_special_tokens=True))
# print('tgt', tokenizer.decode(a.labels[0], skip_special_tokens=True))

# standard with prefix
MODEL_PATH='/ivi/ilps/personal/dju/checkpoints/ctxcomp-flan-t5-arge-inverted-mds-std_prefix/checkpoint-5000'
model = FiDT5.from_pretrained(MODEL_PATH)

print('src', tokenizer.decode(a.input_ids[:, :, 10:100].contiguous().view(-1), skip_special_tokens=True))
for i in range(a.input_ids.size(1)):
    prefix=tokenizer(
        f"<pad>[{i+1}] ", add_special_tokens=False, return_tensors='pt'
    ).input_ids
    output = model.generate(
        input_ids=a.input_ids, 
        attention_mask=a.attention_mask,
        decoder_input_ids=prefix,
        do_sample=True,
        top_p=0.1,
        temperature=0.2,
        min_new_tokens=32,
        max_new_tokens=512,
    )
    print('pred', tokenizer.decode(output[0], skip_special_tokens=True))

print('tgt', tokenizer.decode(a.labels[0], skip_special_tokens=True))
