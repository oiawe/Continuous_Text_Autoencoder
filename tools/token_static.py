import os
import sys
sys.path.append(os.getcwd())

import pandas as pd
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path

from models.tokenizer import TokenizerWrapper

# Configuration parameters
tokenizer_path = 'datas/tokenizer'
data_dir = 'datas/falcon-refinedweb/data'
output_csv = './reports/token_statistics_train.csv'
max_workers = 32


def process_single_file(file_path, tokenizer_path):
    """
    Process a single parquet file and count token occurrences

    Args:
        file_path: Path to parquet file
        tokenizer_path: Path to tokenizer

    Returns:
        Counter: Dictionary mapping token_id -> count
    """
    file_path = Path(file_path)
    tokenizer = TokenizerWrapper(tokenizer_path)
    token_counter = Counter()

    try:
        df = pd.read_parquet(file_path)

        if 'content' not in df.columns:
            print(f"Warning: File {file_path} does not contain 'content' column.")
            return token_counter

        for content in tqdm(df['content'], desc=f"Tokenizing {file_path.name}"):
            if pd.isna(content):
                continue

            # Encode text to token IDs
            token_ids = tokenizer.encode(content).tolist()

            # Count occurrences of each token
            token_counter.update(token_ids)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

    return token_counter


def merge_counters(counter_list):
    """
    Merge multiple Counter objects

    Args:
        counter_list: List of Counter objects

    Returns:
        Counter: Merged Counter object
    """
    merged = Counter()
    for counter in counter_list:
        merged.update(counter)
    return merged


def main():
    # Get all parquet files
    parquet_files = list(Path(data_dir).glob('*.parquet'))
    total_files = len(parquet_files)

    print(f"Found {total_files} parquet files in {data_dir}")
    print(f"Using {max_workers} workers for parallel processing")

    # Store statistics from all files
    all_counters = []

    # Use ProcessPoolExecutor for multiprocessing
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(process_single_file, str(file_path), tokenizer_path): file_path
            for file_path in parquet_files
        }

        # Display progress with tqdm
        with tqdm(total=total_files, desc='Processing parquet files') as pbar:
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    counter = future.result()
                    all_counters.append(counter)
                except Exception as e:
                    print(f"Error getting result for {file_path}: {e}")

                pbar.update(1)

    # Merge all counters
    print("Merging results...")
    final_counter = merge_counters(all_counters)

    # Convert to DataFrame and sort
    print("Creating DataFrame...")
    token_stats = pd.DataFrame([
        {'token_id': token_id, 'count': count}
        for token_id, count in final_counter.items()
    ])

    # Sort by count in descending order
    token_stats = token_stats.sort_values('count', ascending=False).reset_index(drop=True)

    # Display statistics
    print(f"\nTotal unique tokens: {len(token_stats)}")
    print(f"Total token occurrences: {token_stats['count'].sum()}")
    print(f"\nTop 10 most frequent tokens:")
    print(token_stats.head(10))

    # Save to CSV
    print(f"\nSaving to {output_csv}...")
    token_stats.to_csv(output_csv, index=False)
    print(f"Token statistics saved to {output_csv}")


if __name__ == '__main__':
    main()
