import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Union, List
from transformers import Seq2SeqTrainingArguments

@dataclass
class ModelOpt:
    model_name_or_path: Optional[str] = field(default=None)
    config_name: Optional[str] = field(default=None)
    tokenizer_name: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default=None)
    use_fast_tokenizer: bool = field(default=True)
    use_auth_token: bool = field(default=False)
    temperature: Optional[float] = field(default=1)

@dataclass
class DataOpt:
    dataset_config_name: Optional[str] = field(default=None)
    overwrite_cache: bool = field(default=False)
    preprocessing_num_workers: Optional[int] = field(default=None)
    train_file: Optional[str] = field(default=None)
    eval_file: Optional[str] = field(default=None)
    max_src_length: int = field(default=512)
    max_tgt_length: int = field(default=512)

@dataclass
class TrainOpt(Seq2SeqTrainingArguments):
    output_dir: str = field(default='./temp')
    seed: int = field(default=42)
    data_seed: int = field(default=None)
    eval_strategy: Optional[str] = field(default='steps')
    per_device_train_batch_size: int = field(default=2)
    per_device_eval_batch_size: int = field(default=2)
    resume_from_checkpoint: Optional[str] = field(default=None)
    save_total_limit: Optional[int] = field(default=10)
    learning_rate: Union[float] = field(default=1e-5)
    remove_unused_columns: bool = field(default=False)
    report_to: Optional[List[str]] = field(default=None)
    warmup_steps: int = field(default=0)
    max_num_contexts: int = field(default=10)
    num_contexts: int = field(default=None)
    num_distractor_docs: int = field(default=1)
    num_redundant_docs: int = field(default=1)
    logging_steps: int = field(default=10)

    # datacollator
    collator_type: str = field(default='standard')

