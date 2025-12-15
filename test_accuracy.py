import os

import json
import torch
from tqdm import tqdm

from models.model import TextVAE, MODEL_CONFIG
from models.tokenizer import TokenizerWrapper

# Configuration
MODEL_PATH = 'checkpoints/0/checkpoint_10000.pt'
TOKENIZER_PATH = 'datas/tokenizer'
DATA_PATH = 'datas/filtered_falcon_5000.jsonl'
ERROR_OUTPUT_PATH = './reports/error_cases.jsonl'
DEVICE = 'cuda'

def load_model(model_path, device):
    """Load the trained VAE model"""
    model = TextVAE(**MODEL_CONFIG).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def load_data(data_path, max_samples=None):
    """Load data from jsonl file"""
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            data.append(json.loads(line))
    return data

def test_sample(model, tokenizer, sample, device):
    """Test a single sample and return accuracy metrics"""
    original_text = sample['text']
    original_tokens = torch.tensor(sample['token_ids'], dtype=torch.long).unsqueeze(0).to(device)

    # Pad tokens to match downsample ratio
    seq_len = original_tokens.size(1)
    if seq_len % model.downsample_ratio != 0:
        padding_len = model.downsample_ratio - seq_len % model.downsample_ratio
        padding = torch.full((1, padding_len), tokenizer.pad_token_id, dtype=torch.long).to(device)
        padded_tokens = torch.cat([original_tokens, padding], dim=1)
    else:
        padded_tokens = original_tokens
        padding_len = 0

    # Generate tokens
    with torch.inference_mode():
        generated_tokens = model.generate(padded_tokens)

    # Remove padding for comparison
    if padding_len > 0:
        original_tokens = original_tokens[:, :-padding_len] if padding_len < original_tokens.size(1) else original_tokens
        generated_tokens = generated_tokens[:, :-padding_len] if padding_len < generated_tokens.size(1) else generated_tokens

    # Ensure same length for comparison
    min_len = min(original_tokens.size(1), generated_tokens.size(1))
    original_tokens = original_tokens[:, :min_len]
    generated_tokens = generated_tokens[:, :min_len]

    # Calculate token-level accuracy
    correct = (original_tokens == generated_tokens).sum().item()
    total = min_len
    accuracy = correct / total if total > 0 else 0.0

    # Decode texts - decode the actual tokens used for comparison
    original_text_decoded = tokenizer.decode(original_tokens.squeeze(0).cpu())
    generated_text = tokenizer.decode(generated_tokens.squeeze(0).cpu())

    # Prepare result
    result = {
        'correct_tokens': correct,
        'total_tokens': total,
        'accuracy': accuracy,
        'original_text': original_text,  # Keep for reference
        'original_text_decoded': original_text_decoded,  # Actual tokens used
        'generated_text': generated_text,
        'is_correct': accuracy == 1.0
    }

    return result

def main():
    print("Loading tokenizer...")
    tokenizer = TokenizerWrapper(TOKENIZER_PATH)

    print("Loading model...")
    model = load_model(MODEL_PATH, DEVICE)
    model.print_parameters()

    print(f"\nLoading data from {DATA_PATH}...")
    data = load_data(DATA_PATH)
    print(f"Loaded {len(data)} samples")

    # Test all samples
    print("\nTesting samples...")
    all_results = []
    total_correct_tokens = 0
    total_tokens = 0
    perfect_reconstructions = 0
    error_cases = []

    for sample in tqdm(data):
        result = test_sample(model, tokenizer, sample, DEVICE)
        all_results.append(result)

        total_correct_tokens += result['correct_tokens']
        total_tokens += result['total_tokens']

        if result['is_correct']:
            perfect_reconstructions += 1
        else:
            # Save error case
            error_cases.append({
                'original_text_full': result['original_text'],  # Full original text from dataset
                'original_text_decoded': result['original_text_decoded'],  # Decoded from tokens used
                'generated_text': result['generated_text'],
                'token_accuracy': result['accuracy'],
                'correct_tokens': result['correct_tokens'],
                'total_tokens': result['total_tokens']
            })

    # Calculate overall statistics
    overall_token_accuracy = total_correct_tokens / total_tokens if total_tokens > 0 else 0.0
    sequence_accuracy = perfect_reconstructions / len(data) if len(data) > 0 else 0.0

    # Print results
    print("\n" + "="*80)
    print("TEST RESULTS")
    print("="*80)
    print(f"Total samples: {len(data)}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Correct tokens: {total_correct_tokens:,}")
    print(f"\nToken-level accuracy: {overall_token_accuracy:.4f} ({overall_token_accuracy*100:.2f}%)")
    print(f"Sequence-level accuracy: {sequence_accuracy:.4f} ({sequence_accuracy*100:.2f}%)")
    print(f"Perfect reconstructions: {perfect_reconstructions}/{len(data)}")
    print(f"Error cases: {len(error_cases)}/{len(data)}")
    print("="*80)

    # Save error cases
    if error_cases:
        print(f"\nSaving {len(error_cases)} error cases to {ERROR_OUTPUT_PATH}...")
        with open(ERROR_OUTPUT_PATH, 'w', encoding='utf-8') as f:
            for error_case in error_cases:
                f.write(json.dumps(error_case, ensure_ascii=False) + '\n')
        print(f"Error cases saved successfully!")
    else:
        print("\nNo error cases found! All sequences were perfectly reconstructed.")

    # Show a few examples of errors
    if error_cases:
        print("\n" + "="*80)
        print("SAMPLE ERROR CASES (first 3)")
        print("="*80)
        for i, error in enumerate(error_cases[:3]):
            print(f"\n--- Error Case {i+1} ---")
            print(f"Token Accuracy: {error['token_accuracy']:.4f} ({error['correct_tokens']}/{error['total_tokens']})")
            print(f"\nOriginal Text (decoded from tokens, first 200 chars):")
            orig = error['original_text_decoded']
            print(orig[:200] + "..." if len(orig) > 200 else orig)
            print(f"\nGenerated Text (first 200 chars):")
            gen = error['generated_text']
            print(gen[:200] + "..." if len(gen) > 200 else gen)
            print("-" * 80)

if __name__ == "__main__":
    main()
