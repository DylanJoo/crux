import sys
import torch
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    GenerationConfig,
    Seq2SeqTrainer
)
from datasets import load_dataset

# customized modules
from models import FiDT5
# from models.fidt5_comp import FiDT5
from options import ModelOpt, DataOpt, TrainOpt
from utils import update_tokenizer

import os

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class VerboseTrainer(Seq2SeqTrainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        if (self.state.global_step % 10 == 0) and (self.args.should_save):
            kwargs = {
                'input_ids': inputs.input_ids[:1],
                'attention_mask': inputs.attention_mask[:1],
                'max_new_tokens': 512,
                'temperature': 0.7
            }
            if 'decoder_input_ids' in inputs:
                kwargs.update({'decoder_input_ids': inputs.decoder_input_ids[:1]})

            with torch.no_grad():
                output = model.generate(**kwargs)

            logger.info(self.tokenizer.decode(output[0], skip_special_tokens=False))

        return super().compute_loss(
            model=model,
            inputs=inputs,
            return_outputs=return_outputs
        )

def main():
    # Parse argument for huggingface packages
    parser = HfArgumentParser((ModelOpt, DataOpt, TrainOpt))
    model_opt, data_opt, train_opt = parser.parse_args_into_dataclasses()

    # Model
    tokenizer = AutoTokenizer.from_pretrained(model_opt.tokenizer_name)
    model = FiDT5.from_pretrained(model_opt.model_name_or_path)
    # model.use_cache = False
    tokenizer = update_tokenizer(tokenizer)

    ## resizing embeddings (adding new tokens is recommended (instead of preserved tokens))
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Dataset
    dataset = load_dataset('json', data_files=data_opt.train_file, keep_in_memory=True)
    dataset = dataset.filter(lambda x: len(x['docids']) !=0 )
    n_examples = len(dataset['train'])
    if train_opt.do_eval:
        dataset = dataset['train'].train_test_split(test_size=100, seed=1997)

    # Datacollator
    from data import Standard, StandardWithPrefix
    collator_class = {
        "standard": Standard, 
        "standard_with_prefix": StandardWithPrefix
    }[train_opt.collator_type]

    data_collator = collator_class(
        tokenizer=tokenizer, 
        max_src_length=data_opt.max_src_length,
        max_tgt_length=data_opt.max_tgt_length,
        max_num_contexts=train_opt.max_num_contexts,
        num_distractor_docs=train_opt.num_distractor_docs,
        num_redundant_docs=train_opt.num_redundant_docs,
        shuffle=True,
    )
    # num_contexts is for batch-wsie training

    # Trainer
    trainer = VerboseTrainer(
        model=model, 
        args=train_opt,
        tokenizer=tokenizer,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'] if train_opt.do_eval else None,
        data_collator=data_collator,
    )
    results = trainer.train()

if __name__ == '__main__':
    main()
