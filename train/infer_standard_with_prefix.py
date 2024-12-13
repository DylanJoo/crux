import gc
import torch
import json
import random
from tqdm import tqdm

device='cuda'
# device='cpu'
from models import FiDT5
MODEL_PATH='/ivi/ilps/personal/dju/checkpoints/ctxcomp-flan-t5-large-inverted-mds-std_prefix/checkpoint-5000'
model = FiDT5.from_pretrained(MODEL_PATH).to(device)

from transformers import AutoTokenizer
from utils import update_tokenizer
tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-large')
tokenizer = update_tokenizer(tokenizer)

from datasets import load_dataset
dataset = load_dataset('json', data_files='/home/dju/rag-rerank/data/inverted-mds/test.jsonl', keep_in_memory=True)['train']
random.seed(10)
dataset = dataset.select(
    random.sample(range(len(dataset)), 100)
)

from data import Standard
collator = Standard(
    tokenizer=tokenizer,
    max_src_length=1023,
    max_tgt_length=512,
    n_contexts=None,
    shuffle=False # we use the same order
)

writer = open('test_standard_with_prefix.json', 'w')
for example in tqdm(dataset):
    topic = example['question']
    input = collator([example]).to(device)

    labels = ""
    prediction = ""
    for i, summary in enumerate(example['comp_ctxs']):
        labels += f"[{i+1}] example['comp_ctxs'][0]"
        prefix = tokenizer(f"<pad>[{i+1}] ", return_tensors='pt').input_ids.to(device)
        with torch.no_grad():
            output = model.generate(
                input_ids=input.input_ids, 
                attention_mask=input.attention_mask,
                decoder_input_ids=prefix,
                do_sample=True,
                top_p=0.1,
                temperature=0.2,
                min_new_tokens=32,
                max_new_tokens=512,
            )
            prediction += tokenizer.decode(output[0], skip_special_tokens=True)

    labels = tokenizer.decode(input.labels[0], skip_special_tokens=True)
    writer.write(json.dumps(
        {"topic": topic,"prediction": prediction, "label": labels}, 
        indent=4)+'\n'
    )
    del output
    torch.cuda.empty_cache()
    gc.collect()

writer.close()
