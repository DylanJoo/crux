""" [TODO] retrofit the gtr encoders into instruction-tunned dense retriever
"""
import torch
import copy
from transformers import (
    T5EncoderModel, 
    T5Config
)
from transformers.models.t5.modeling_t5 import T5Stack
import torch.nn as nn
import torch.nn.functional as F

class GTREncoder(T5EncoderModel):
    def __init__(self, config: T5Config):
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)
        self.linear = nn.Linear(config.d_model, config.d_model, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def forward_start(self, 
                      q_inputs, qs_inputs, p_inputs, 
                      document_encoder=None, **kwargs):
        """ See the customized datacollator in `data.py` for detail.  """
        # 1) Get representation 
        ## document embeddings
        ### Setting 0: freezed document encoder
        if document_encoder is not None:
            D = document_encoder.encode(p_inputs, normalized=False)
        else:
            # [NOTE] To simplify the PoC, this is deprecated so far.
            d_output = self.forward(**p_inputs)
            D = self.mean_pooling(d_output, p_inputs['attention_mask'])
            D = self.linear(D)

        ## query embeddings
        q_output = self.forward(**q_inputs)
        Q = self.mean_pooling(q_output, q_inputs['attention_mask'])
        Q = self.linear(Q)

        ## statement-awared query embeddings
        qs_output = self.forward(**qs_inputs)
        QS = self.mean_pooling(qs_output, qs_inputs['attention_mask'])
        QS = self.linear(QS)

        ## query-passage relevance logits
        qp_logits = Q @ D.transpose(0, 1)
        qsp_logits = QS @ D.transpose(0, 1)

        return {'qp_logits': qp_logits, 'qsp_logits': qsp_logits}

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @torch.no_grad()
    def encode(self, inputs, normalized=True, projected=True):
        model_output = self.forward(**inputs)
        embeddings = self.mean_pooling(model_output, inputs['attention_mask'])

        # linear projection for retrieval
        if projected:
            embeddings = self.linear(embeddings)

        # normalized for retrieval
        if normalized:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings
