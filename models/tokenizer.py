import torch

DEFAULT_TOKENIZER_PATH = 'answerdotai/ModernBERT-base'

def get_pretrained_tokenizer(tokenizer_path):
    from transformers import AutoTokenizer
    tokenizer_path = tokenizer_path if tokenizer_path else DEFAULT_TOKENIZER_PATH
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return tokenizer

class TokenizerWrapper:
    def __init__(self, tokenizer_path):
        self.tokenizer = get_pretrained_tokenizer(tokenizer_path)
        self.pad_token_id = self.tokenizer.pad_token_id
        self.vocab_size = 50368

    def encode(self, text):
        return self.tokenizer.encode(text, return_tensors='pt').squeeze(0)[:-1] # discard [SEP] token at the end

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

if __name__ == '__main__':
    tokenizer_path = 'datas/tokenizer'
    tokenizer = TokenizerWrapper(tokenizer_path)

    text = "The capital of France is."
    text = 'aaa bbb ccc'
    inputs = tokenizer.encode(text)
    print(inputs.shape)
    print(inputs)
    inputs = torch.nn.functional.pad(inputs, (0, 20 - inputs.shape[0]), value=tokenizer.pad_token_id)
    outputs = tokenizer.decode(inputs)
    print(outputs)