import os
import sys
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
import json
from tqdm import tqdm

from models.tokenizer import TokenizerWrapper

# Configuration
tokenizer_path = 'datas/tokenizer'
input_parquet = 'datas/test_falcon/train-00050-of-05534-db9cb02b30b9febe.parquet'
token_stats_file = 'reports/token_statistics_train.csv'
output_file = 'datas/filtered_falcon_5000.jsonl'
target_count = 5000
target_length = 512

# Initialize tokenizer
print("Loading tokenizer...")
tokenizer = TokenizerWrapper(tokenizer_path)

# Load token statistics
print("Loading token statistics...")
token_stats = pd.read_csv(token_stats_file)
print(f"Total tokens in vocabulary: {len(token_stats)}")

# Calculate the threshold for the lowest 10% frequency
# Sort by count to find the 10th percentile
token_stats_sorted = token_stats.sort_values('count')
threshold_idx = int(len(token_stats) * 0.1)
low_freq_threshold = token_stats_sorted.iloc[threshold_idx]['count']
print(f"Low frequency threshold (10th percentile): {low_freq_threshold}")

# Create set of low frequency tokens and valid tokens
low_freq_tokens = set(token_stats_sorted[token_stats_sorted['count'] <= low_freq_threshold]['token_id'].values)
valid_tokens = set(token_stats['token_id'].values)
print(f"Number of low frequency tokens (lowest 10%): {len(low_freq_tokens)}")
print(f"Total valid tokens: {len(valid_tokens)}")

# Load input parquet file
print(f"Loading parquet file: {input_parquet}")
df = pd.read_parquet(input_parquet)
print(f"Total records in parquet: {len(df)}")

if 'content' not in df.columns:
    raise ValueError(f"Parquet file does not contain 'content' column. Available columns: {df.columns.tolist()}")

# Process texts
filtered_texts = []
filtered_token_ids = []

print(f"Processing texts to collect {target_count} valid samples...")
for idx, content in enumerate(tqdm(df['content'], desc='Processing texts')):
    if len(filtered_texts) >= target_count:
        break

    try:
        # Tokenize the content
        token_ids = tokenizer.encode(content).tolist()

        # Rule 1: Discard if length < 512
        if len(token_ids) < target_length:
            continue

        # Rule 2: Randomly truncate to 512 if length > 512
        if len(token_ids) > target_length:
            # Random starting position for truncation
            max_start = len(token_ids) - target_length
            start_idx = np.random.randint(0, max_start + 1)
            token_ids = token_ids[start_idx:start_idx + target_length]

        # Convert to set for faster lookup
        token_set = set(token_ids)

        # Rule 3: Discard if contains tokens not in valid tokens (never appeared)
        if not token_set.issubset(valid_tokens):
            continue

        # Rule 4: Discard if contains low frequency tokens (lowest 10%)
        if token_set.intersection(low_freq_tokens):
            continue

        # If all rules pass, add to filtered results
        filtered_texts.append(content)
        filtered_token_ids.append(token_ids)

    except Exception as e:
        # Skip if there's any error in processing
        continue

print(f"Successfully collected {len(filtered_texts)} valid samples")

if len(filtered_texts) < target_count:
    print(f"Warning: Only found {len(filtered_texts)} samples, less than target {target_count}")

# Save to JSONL
print(f"Saving results to {output_file}...")
with open(output_file, 'w', encoding='utf-8') as f:
    for text, token_ids in zip(filtered_texts, filtered_token_ids):
        json_obj = {
            'text': text,
            'token_ids': token_ids
        }
        f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
print(f"Done! Saved {len(filtered_texts)} samples to {output_file}")

# Print some statistics
print("\nStatistics:")
print(f"Total texts processed: {idx + 1}")
print(f"Valid samples collected: {len(filtered_texts)}")
print(f"Success rate: {len(filtered_texts) / (idx + 1) * 100:.2f}%")
