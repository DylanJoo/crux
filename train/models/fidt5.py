import copy
import torch
from transformers import T5ForConditionalGeneration, T5Config
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from typing import Optional, Tuple, Union
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
from transformers.models.t5.modeling_t5 import (
    T5Stack, 
    T5Block, 
    T5LayerSelfAttention, 
    T5LayerCrossAttention, 
    T5Attention, 
    T5LayerNorm,
    T5LayerFF
)

class FiDT5(T5ForConditionalGeneration):
    def __init__(self, config: T5Config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = FiDT5EncoderStack(encoder_config, self.shared) # replace 

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = FiDT5DecoderStack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def get_crossattention_scores(self, context_mask):
        raise NotImplementedError('Please implement this function.')

class FiDT5EncoderStack(T5Stack):

    def forward(
        self, 
        input_ids, 
        attention_mask, 
        past_key_values=None, 
        **kwargs
    ):
        """ Wrap/unwrap input/ouput with this class (replace t5-encoder) 

        :param input_ids: the tokenized input ids with shape (BN, L)
        :param attention_mask: the attention mask with shape (B, NL)
        :param context_embeds: 
            the statement-aware query embeddings with shape (B, M); each embedding
            was pre-encoded (so far) tensors with GTREncoder.

        :return encoder_outputs: the huggingface model output class.
        """
        if input_ids.dim() == 3: # normal usage of FiD
            B, N, L = input_ids.size()
        else:
            B, L = input_ids.size()
            N = 1

        # Modifying 1
        ## For `input_ids`, 
        ## transform from original batch into enuemrated batch.
        ## i.e. from (B, N, L) to (BN, L) 
        input_ids = input_ids.view(B*N, -1)
        ## For `attention_mask`, 
        ## transform from original batch into enuemrated batch.
        ## i.e. from (B, NL) to (BN, L) 
        attention_mask = attention_mask.view(B*N, -1)

        # Minor modifying
        ## Prefix tuning at decoder. Avoid `past_key_value` being considered.
        encoder_outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask, 
            past_key_values=None,
            **kwargs
        )

        # Modifying 2
        ## transform from enuemrated batch into original batch 
        ## I.e. from (BN, L, H) to (B, NL, H) 
        encoder_outputs['last_hidden_state'] = \
                encoder_outputs['last_hidden_state'].view(B, N*L, -1)

        return encoder_outputs

class FiDT5DecoderStack(T5Stack):
    """ adopt the relative attention (self & encdec) at the first (index=0) layer.  """
    def __init__(self, config, embed_tokens=None):
        super().__init__(config)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        self.block = nn.ModuleList(
            [FiDT5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        # Initialize weights and apply final processing
        self.post_init()
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

class FiDT5Block(T5Block):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__(config)
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(T5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        if self.is_decoder:
            self.layer.append(FiDT5LayerCrossAttention(config, has_relative_attention_bias))

        self.layer.append(T5LayerFF(config))

class FiDT5LayerCrossAttention(T5LayerCrossAttention):
    """ default relative attention of CrossAttn is always set False. """
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__(config)
        self.EncDecAttention = T5Attention(config, has_relative_attention_bias=has_relative_attention_bias)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)
