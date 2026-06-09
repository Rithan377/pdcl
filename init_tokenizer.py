"""
Initialize BPE Tokenizer from Training Data
=============================================
Trains a BPE tokenizer on the training dataset and saves it.
"""

import json
from pdcl_tokenizer import BPETokenizer

print("\n" + "="*70)
print("  INITIALIZING BPE TOKENIZER")
print("="*70 + "\n")

# Load training data
print("Loading training data...")
with open('./data/train.json') as f:
    train_data = json.load(f)

print(f"  ✓ Loaded {len(train_data)} training samples")

# Extract corpus (questions + documents)
corpus = []
for sample in train_data:
    corpus.append(sample['question'])
    corpus.append(sample['document'])

print(f"  ✓ Extracted {len(corpus)} text chunks")

# Train tokenizer
print(f"\nTraining BPE tokenizer (vocab_size=8000)...")
tokenizer = BPETokenizer()
tokenizer.train(corpus, vocab_size=8000, min_freq=2)

print(f"  ✓ Vocabulary size: {tokenizer.vocab_size}")
print(f"  ✓ Learned {len(tokenizer.merges)} merge rules")

# Save tokenizer
print(f"\nSaving tokenizer to ./tokenizer...")
tokenizer.save('./tokenizer')
print(f"  ✓ Tokenizer saved")

print("\n" + "="*70)
print("  TOKENIZER READY")
print("="*70 + "\n")
