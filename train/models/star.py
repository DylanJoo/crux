from transformers import AutoModel, AutoTokenizer
from gtr import GTREncoder

class StarEmbeds:
    def __init__(self, encoder_name_or_path=None, device='cpu', **kwargs):
        self.encoder = GTREncoder.from_pretrained(encoder_name_or_path)
        self.encoder.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name_or_path)

    def get_embeddings(self, queries, statements, max_seq=64):
        tokenized_inputs = self.tokenizer(
                [f"{q} </s> {s}" for (q, s) in zip(queries, statements)],
                max_length=max_seq,
                truncation=True,
                padding=True,
                return_tensors='pt'
        ).to(self.encoder.device)

        embeddings = self.encoder.encode(
                tokenized_inputs, 
                normalized=False, projected=False
        )
        return embeddings

