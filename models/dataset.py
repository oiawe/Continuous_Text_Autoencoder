import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random

import pandas as pd

import torch
from torch.utils.data import Dataset

from models.tokenizer import TokenizerWrapper

class ParquetContentDataset(Dataset):
    def __init__(self, parquet_path, tokenizer_path, chunk_size):
        df = pd.read_parquet(parquet_path)

        self.contents = df["content"].tolist()
        self.tokenizer = TokenizerWrapper(tokenizer_path)
        self.padding_value = self.tokenizer.pad_token_id

        self.chunk_size = chunk_size

    def __len__(self):
        return len(self.contents)

    def random_chunk(self, token_ids):
        seq_len = token_ids.size(-1)

        if seq_len >= self.chunk_size:
            start = random.randint(0, seq_len - self.chunk_size)
            return token_ids[..., start : start + self.chunk_size]

        padded = torch.nn.functional.pad(token_ids, (0, self.chunk_size - seq_len), value=self.padding_value)
        return padded

    def __getitem__(self, idx):
        text = self.contents[idx]
        token_ids = self.tokenizer.encode(text)
        token_ids = self.random_chunk(token_ids)
        return token_ids

if __name__ == '__main__':
    dataset = ParquetContentDataset(
        parquet_path='datas/falcon_data/data/train_sample_100.parquet',
        tokenizer_path='datas/tokenizer',
        chunk_size=128
    )

    print(f"Dataset size: {len(dataset)}")

    for i in range(3):
        token_ids = dataset[i]
        print(f"Sample {i} token IDs shape: {token_ids.shape}")
        print(token_ids)

    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for batch_text in dataloader:
        print(batch_text)
        print(batch_text.shape)
        break