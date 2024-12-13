import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers import (
    T5ForConditionalGeneration,
    AutoModelForSequenceClassification
)

class monoT5(nn.Module):

    def __init__(
        self, 
        model_name_or_dir='castorini/monot5-3b-msmarco-10k',
        tokenizer_name=None, 
        device='auto', 
        fp16=False
    ):
        """ monot5 will use the logit of the first token.
        Then estimate the probablity re-softmax 'yes' and 'no' tokens.  
        """
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_name_or_dir,
            torch_dtype=torch.float16 if fp16 else torch.float32,
            device_map=device
        )
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name or model_name_or_dir
        )
        self.REL = self.tokenizer.encode('yes')[0]
        self.NREL = self.tokenizer.encode('no')[0]

    def predict(self, queries, documents, titles=None, max_length=384):
        # prepare inputs
        # [NOTE] sometimes the token `relevant` token will be truncated
        if titles is not None:
            documents = [f'{title} {text}'.strip() for title, text in zip(titles, documents)] 
        text_pairs = [f"Query: {q} Document: {d} Relevant:" for (q, d) in zip(queries, documents)]
        inputs = self.tokenizer(
            text_pairs,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=max_length
        ).to(self.model.device)

        # generate first token 
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=2,
            return_dict_in_generate=True,
            output_scores=True
        )

        # scoring the first logits and apply softmax
        scores = outputs.scores[0]
        scores = scores[:, [self.NREL, self.REL]]
        scores = F.log_softmax(scores, dim=-1)
        scores = scores[:, 1].exp().cpu().detach().numpy()
        return scores


class monoBERT(nn.Module):

    def __init__(
        self,
        model_name_or_dir='unicamp-dl/mMiniLM-L6-v2-mmarco-v2',
        tokenizer_name=None, 
        device='auto', 
        fp16=False
    ):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_dir,
            torch_dtype=torch.float16 if fp16 else torch.float32,
            device_map=device,
        )
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name or model_name_or_dir
        )

    def predict(self, queries, documents, titles=None, max_length=384):
        # prepare inputs
        if titles is not None:
            documents = [f'{title} {text}'.strip() for title, text in zip(titles, documents)] 
        inputs = self.tokenizer(
                queries, documents,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=max_length
        ).to(self.model.device)

        # predict relevance
        scores = self.model(**inputs).logits

        if self.model.config.num_labels == 1:
            scores = scores.squeeze().cpu().detach().numpy()
        else:
            scores = scores[:, 1].cpu().detach().numpy()
        return scores
