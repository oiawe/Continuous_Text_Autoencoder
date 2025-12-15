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
token_stats_file = 'reports/token_statistics_train.csv'
output_file = 'datas/filtered_random_5000.jsonl'
target_count = 5000
target_length = 512

# Set random seed for reproducibility
np.random.seed(42)

# Initialize tokenizer
print("Loading tokenizer...")
tokenizer = TokenizerWrapper(tokenizer_path)

# Load token statistics
print("Loading token statistics...")
token_stats = pd.read_csv(token_stats_file)
print(f"Total tokens in vocabulary: {len(token_stats)}")

# Calculate the threshold for the top 90% frequency (exclude lowest 10%)
token_stats_sorted = token_stats.sort_values('count')
threshold_idx = int(len(token_stats) * 0.1)
low_freq_threshold = token_stats_sorted.iloc[threshold_idx]['count']
print(f"Low frequency threshold (10th percentile): {low_freq_threshold}")

# Get tokens with frequency in top 90% (exclude lowest 10%)
high_freq_tokens = token_stats[token_stats['count'] > low_freq_threshold]['token_id'].values
print(f"Number of high frequency tokens (top 90%): {len(high_freq_tokens)}")

# Generate random token sequences
print(f"Generating {target_count} random token sequences...")
generated_texts = []
generated_token_ids = []

for i in tqdm(range(target_count), desc='Generating sequences'):
    # Randomly sample 512 tokens from high frequency tokens
    random_token_ids = np.random.choice(high_freq_tokens, size=target_length, replace=True).tolist()

    try:
        # Decode token IDs to text
        decoded_text = tokenizer.decode(random_token_ids)

        # Add to results
        generated_texts.append(decoded_text)
        generated_token_ids.append(random_token_ids)
    except Exception as e:
        print(f"Error decoding sequence {i}: {e}")
        continue

print(f"Successfully generated {len(generated_texts)} sequences")

# Save to JSONL
print(f"Saving results to {output_file}...")
with open(output_file, 'w', encoding='utf-8') as f:
    for text, token_ids in zip(generated_texts, generated_token_ids):
        json_obj = {
            'text': text,
            'token_ids': token_ids
        }
        f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
print(f"Done! Saved {len(generated_texts)} samples to {output_file}")

# Print some statistics
print("\nStatistics:")
print(f"Target sequences: {target_count}")
print(f"Successfully generated: {len(generated_texts)}")
print(f"Sequence length: {target_length}")
print(f"Vocabulary size used: {len(high_freq_tokens)}")

# Print a sample
if len(generated_texts) > 0:
    print("\nSample generated text:")
    print(f"Token IDs: {generated_token_ids[0][:20]}...")  # First 20 tokens
    print(f"Text: {generated_texts[0][:200]}...")  # First 200 characters
