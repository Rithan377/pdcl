"""
PDCL BPE Tokenizer — Built from Raw Math
==========================================
Byte Pair Encoding from scratch.
No libraries. No pretrained vocab.
Works on ANY text — noisy, clean, unseen.

Core Algorithm:
1. Start with every character as its own token
2. Count all adjacent pairs in corpus
3. Merge the most frequent pair into a new token
4. Repeat N times to build vocabulary
5. Encode new text using learned merges

Math:
- Pair frequency: freq(a,b) = count of (a,b) adjacent in all words
- Merge rule: replace all (a,b) → (ab) across corpus
- Encode: greedily apply learned merges in order
"""

import json
import os
import re
import collections
from typing import Dict, List, Tuple, Optional


# ─────────────────────────────────────────────
# CORE BPE MATH
# ─────────────────────────────────────────────

def get_word_freqs(corpus: List[str]) -> Dict[Tuple, int]:
    """
    Step 1 — Convert corpus to word frequency dictionary.

    Each word is split into characters with </w> end marker.
    End marker lets the algorithm learn word boundaries.

    Example:
        "hello world hello" ->
        {('h','e','l','l','o','</w>'): 2,
         ('w','o','r','l','d','</w>'): 1}
    """
    word_freqs = collections.defaultdict(int)
    for text in corpus:
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text.lower().strip())
        for word in text.split():
            # Clean word — keep alphanumeric and basic punctuation
            word = re.sub(r'[^\w\$\%\.\,\-]', '', word)
            if len(word) == 0:
                continue
            # Split into characters + end of word marker
            chars = tuple(list(word) + ['</w>'])
            word_freqs[chars] += 1
    return dict(word_freqs)


def get_pair_freqs(word_freqs: Dict[Tuple, int]) -> Dict[Tuple, int]:
    """
    Step 2 — Count frequency of every adjacent pair across all words.

    Math:
        For word W = (c1, c2, c3, ..., cn) with frequency f:
        pair_freq[(c1,c2)] += f
        pair_freq[(c2,c3)] += f
        ...

    This is the core BPE counting step.
    """
    pair_freqs = collections.defaultdict(int)
    for word, freq in word_freqs.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_freqs[pair] += freq
    return dict(pair_freqs)


def merge_pair(pair: Tuple[str, str],
               word_freqs: Dict[Tuple, int]) -> Dict[Tuple, int]:
    """
    Step 3 — Merge the most frequent pair across all words.

    Math:
        For each word W containing (a, b):
        Replace every occurrence of (a, b) with (ab)

    Example:
        pair = ('e', 's')
        ('t','h','e','s','e','</w>') -> ('t','h','es','e','</w>')
    """
    new_word_freqs = {}
    merged = ''.join(pair)  # e.g. ('e','s') -> 'es'

    for word, freq in word_freqs.items():
        new_word = []
        i = 0
        while i < len(word):
            # Check if current and next token form the target pair
            if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                new_word.append(merged)
                i += 2  # Skip both tokens — they are now merged
            else:
                new_word.append(word[i])
                i += 1
        new_word_freqs[tuple(new_word)] = freq

    return new_word_freqs


def learn_bpe(corpus: List[str],
              vocab_size: int = 8000,
              min_freq: int = 2) -> Tuple[Dict, List]:
    """
    Full BPE training loop.

    Algorithm:
        1. Initialize vocab with all unique characters
        2. Repeat vocab_size times:
            a. Count all adjacent pairs
            b. Find most frequent pair
            c. Merge that pair everywhere
            d. Add merged token to vocabulary
        3. Return vocabulary and ordered merge rules

    vocab_size controls how many merges we learn.
    Higher = larger vocabulary = better coverage but more memory.
    """
    print(f"Building word frequencies from corpus...")
    word_freqs = get_word_freqs(corpus)
    print(f"  Unique words in corpus: {len(word_freqs)}")

    # Initial vocabulary = all unique characters
    vocab = set()
    for word in word_freqs:
        for char in word:
            vocab.add(char)
    vocab.add('</w>')
    vocab.add('<unk>')   # Unknown token
    vocab.add('<pad>')   # Padding token
    vocab.add('<sep>')   # Separator token
    vocab.add('<ans>')   # Answer start token — useful for QA task
    vocab.add('<doc>')   # Document start token
    vocab.add('<que>')   # Question start token

    print(f"  Initial character vocab size: {len(vocab)}")

    merges = []  # Ordered list of merge rules — ORDER MATTERS for encoding
    num_merges = vocab_size - len(vocab)
    print(f"  Learning {num_merges} BPE merges...")

    for i in range(num_merges):
        # Count all pairs
        pair_freqs = get_pair_freqs(word_freqs)

        if not pair_freqs:
            print(f"  No more pairs to merge at step {i}")
            break

        # Find most frequent pair
        best_pair = max(pair_freqs, key=pair_freqs.get)
        best_freq = pair_freqs[best_pair]

        # Stop if best pair appears less than min_freq times
        if best_freq < min_freq:
            print(f"  Stopping: best pair frequency {best_freq} < min_freq {min_freq}")
            break

        # Merge that pair everywhere
        word_freqs = merge_pair(best_pair, word_freqs)

        # Add merged token to vocab
        merged_token = ''.join(best_pair)
        vocab.add(merged_token)
        merges.append(best_pair)

        if (i + 1) % 500 == 0:
            print(f"  Merge {i+1}/{num_merges} | best: {''.join(best_pair)} (freq={best_freq}) | vocab={len(vocab)}")

    print(f"  Final vocab size: {len(vocab)}")
    return vocab, merges


# ─────────────────────────────────────────────
# ENCODER — apply learned BPE merges
# ─────────────────────────────────────────────

# Global cache to optimize BPE encoding speed (reduces repeated merge loops for common words)
_encode_word_cache = {}

def encode_word(word: str, merges: List[Tuple]) -> List[str]:
    """
    Encode a single word using learned merge rules.

    Algorithm:
        1. Split word into characters
        2. Apply each merge rule in learned order
        3. Return list of subword tokens

    Key insight: merges must be applied IN ORDER they were learned.
    Earlier merges = more frequent = applied first.
    """
    if len(word) == 0:
        return []

    # Start with character-level split
    word_clean = re.sub(r'[^\w\$\%\.\,\-]', '', word.lower())
    if len(word_clean) == 0:
        return ['<unk>']

    # Check cache first
    if word_clean in _encode_word_cache:
        return _encode_word_cache[word_clean]

    tokens = list(word_clean) + ['</w>']

    # Apply each merge rule in order
    for pair in merges:
        i = 0
        new_tokens = []
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i+1] == pair[1]:
                new_tokens.append(''.join(pair))
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens

    # Store in cache
    _encode_word_cache[word_clean] = tokens
    return tokens


def encode_text(text: str,
                merges: List[Tuple],
                token2id: Dict[str, int],
                max_length: Optional[int] = None) -> Tuple[List[int], List[int]]:
    """
    Encode full text into token IDs with position tracking.

    Returns:
        token_ids   : list of integer token IDs
        positions   : list of character positions in original text
                      This gives PDCL natural positional information.

    Math:
        token_id[t] = token2id.get(t, token2id['<unk>'])
        position[t] = character offset of token t in original text
    """
    text_norm = re.sub(r'\s+', ' ', text.lower().strip())
    words = text_norm.split()

    token_ids = []
    positions = []
    char_pos = 0

    for word in words:
        word_tokens = encode_word(word, merges)
        for token in word_tokens:
            tid = token2id.get(token, token2id.get('<unk>', 1))
            token_ids.append(tid)
            positions.append(char_pos)
        char_pos += len(word) + 1  # +1 for space

    # Truncate or pad to max_length
    if max_length is not None:
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
            positions = positions[:max_length]
        else:
            pad_id = token2id.get('<pad>', 0)
            pad_len = max_length - len(token_ids)
            token_ids = token_ids + [pad_id] * pad_len
            positions = positions + [0] * pad_len

    return token_ids, positions


# ─────────────────────────────────────────────
# VOCABULARY BUILDER
# ─────────────────────────────────────────────

def build_vocab_mappings(vocab: set) -> Tuple[Dict, Dict]:
    """
    Build token <-> ID mappings.

    Special tokens get reserved IDs 0-6.
    All other tokens sorted for determinism.
    """
    special_tokens = ['<pad>', '<unk>', '<sep>', '<ans>', '<doc>', '<que>', '</w>']

    # Start with special tokens at fixed positions
    token2id = {}
    id2token = {}

    for i, tok in enumerate(special_tokens):
        token2id[tok] = i
        id2token[i] = tok

    # Add remaining vocab sorted (deterministic)
    regular_tokens = sorted(vocab - set(special_tokens))
    for tok in regular_tokens:
        idx = len(token2id)
        token2id[tok] = idx
        id2token[idx] = tok

    return token2id, id2token


# ─────────────────────────────────────────────
# TOKENIZER CLASS
# ─────────────────────────────────────────────

class BPETokenizer:
    """
    Complete BPE Tokenizer for PDCL.
    No external dependencies. Pure math.
    """

    def __init__(self):
        self.vocab = set()
        self.merges = []
        self.token2id = {}
        self.id2token = {}
        self.vocab_size = 0
        self.trained = False

    def train(self, corpus: List[str],
              vocab_size: int = 8000,
              min_freq: int = 2):
        """
        Train BPE on corpus.
        corpus: list of raw text strings.
        """
        print(f"\n{'='*50}")
        print(f"TRAINING BPE TOKENIZER")
        print(f"Target vocab size : {vocab_size}")
        print(f"Min pair frequency: {min_freq}")
        print(f"Corpus size       : {len(corpus)} documents")
        print(f"{'='*50}")

        self.vocab, self.merges = learn_bpe(corpus, vocab_size, min_freq)
        self.token2id, self.id2token = build_vocab_mappings(self.vocab)
        self.vocab_size = len(self.token2id)
        self.trained = True

        print(f"\nTokenizer trained.")
        print(f"Final vocab size  : {self.vocab_size}")
        print(f"Total merge rules : {len(self.merges)}")

    def encode(self,
               text: str,
               max_length: Optional[int] = None) -> Tuple[List[int], List[int]]:
        """
        Encode text to (token_ids, positions).
        Handles unseen words via subword fallback.
        """
        assert self.trained, "Tokenizer must be trained before encoding."
        return encode_text(text, self.merges, self.token2id, max_length)

    def encode_qa(self,
                  document: str,
                  question: str,
                  max_doc_length: int = 2048,
                  max_q_length: int = 64) -> Dict:
        """
        Encode a document + question pair for PDCL.

        Format:
            [<doc>] document tokens [<sep>] [<que>] question tokens

        Returns full encoding with positions for PDCL dimension segmentation.
        """
        doc_id = self.token2id['<doc>']
        que_id = self.token2id['<que>']
        sep_id = self.token2id['<sep>']

        doc_tokens, doc_positions = self.encode(document, max_doc_length)
        que_tokens, que_positions = self.encode(question, max_q_length)

        # Combine: <doc> doc_tokens <sep> <que> que_tokens
        full_tokens = [doc_id] + doc_tokens + [sep_id] + [que_id] + que_tokens
        full_positions = [0] + doc_positions + [0] + [0] + que_positions

        # Segment mask — 0 = document, 1 = question
        # PDCL uses this to know which dimension segment gets which part
        segment_mask = (
            [0] +                          # <doc> token
            [0] * len(doc_tokens) +        # document
            [0] +                          # <sep>
            [1] +                          # <que> token
            [1] * len(que_tokens)          # question
        )

        return {
            'token_ids'    : full_tokens,
            'positions'    : full_positions,
            'segment_mask' : segment_mask,
            'doc_length'   : len(doc_tokens),
            'que_length'   : len(que_tokens),
            'total_length' : len(full_tokens),
        }

    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to text.
        Reverses BPE merges to reconstruct words.
        """
        words = []
        curr_word = ""
        special_tokens = {'<doc>', '<que>', '<sep>', '<pad>', '<unk>'}
        
        for tid in token_ids:
            token = self.id2token.get(tid, '<unk>')
            if token in special_tokens:
                if curr_word:
                    words.append(curr_word)
                    curr_word = ""
                continue
                
            if token.endswith('</w>'):
                curr_word += token[:-4]
                words.append(curr_word)
                curr_word = ""
            else:
                curr_word += token
                
        if curr_word:
            words.append(curr_word)
            
        return ' '.join(words).strip()

    def save(self, path: str):
        """Save tokenizer state to disk."""
        os.makedirs(path, exist_ok=True)
        state = {
            'vocab'     : list(self.vocab),
            'merges'    : [list(m) for m in self.merges],
            'token2id'  : self.token2id,
            'id2token'  : {int(k): v for k, v in self.id2token.items()},
            'vocab_size': self.vocab_size,
        }
        with open(os.path.join(path, 'tokenizer.json'), 'w') as f:
            json.dump(state, f)
        print(f"Tokenizer saved to {path}")

    def load(self, path: str):
        """Load tokenizer from disk."""
        with open(os.path.join(path, 'tokenizer.json')) as f:
            state = json.load(f)
        self.vocab     = set(state['vocab'])
        self.merges    = [tuple(m) for m in state['merges']]
        self.token2id  = state['token2id']
        self.id2token  = {int(k): v for k, v in state['id2token'].items()}
        self.vocab_size = state['vocab_size']
        self.trained   = True
        print(f"Tokenizer loaded from {path}")
        print(f"Vocab size: {self.vocab_size} | Merges: {len(self.merges)}")


# ─────────────────────────────────────────────
# MAIN — Train and verify
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import time

    # Load training data
    print("Loading training corpus...")
    with open('./data/train.json') as f:
        train_data = json.load(f)

    # Build corpus — documents + questions
    corpus = []
    for sample in train_data:
        corpus.append(sample['document'])
        corpus.append(sample['question'])

    print(f"Corpus: {len(corpus)} texts")

    # Train tokenizer
    t0 = time.time()
    tokenizer = BPETokenizer()
    tokenizer.train(corpus, vocab_size=4000, min_freq=2)
    print(f"\nTraining time: {time.time() - t0:.1f}s")

    # Save
    tokenizer.save('./tokenizer')

    # ── VERIFICATION TESTS ──
    print(f"\n{'='*50}")
    print("VERIFICATION TESTS")
    print(f"{'='*50}")

    # Test 1 — Clean text
    test_clean = "The revenue for Q3 2022 reached 5,000,000 dollars."
    ids, pos = tokenizer.encode(test_clean)
    decoded = tokenizer.decode(ids)
    print(f"\nTest 1 — Clean text:")
    print(f"  Input   : {test_clean}")
    print(f"  Tokens  : {len(ids)}")
    print(f"  Decoded : {decoded[:80]}")

    # Test 2 — Noisy text (spelling errors, OCR artifacts)
    test_noisy = "Th3 revnue fr Q3 2022 reachd 5,0O0,000 d0llars accordng t0 the reprt."
    ids2, pos2 = tokenizer.encode(test_noisy)
    decoded2 = tokenizer.decode(ids2)
    print(f"\nTest 2 — Noisy text:")
    print(f"  Input   : {test_noisy}")
    print(f"  Tokens  : {len(ids2)}")
    print(f"  Decoded : {decoded2[:80]}")

    # Test 3 — Completely unseen word
    test_unseen = "xkqzplm florbigated the zyphonic nexuscore."
    ids3, pos3 = tokenizer.encode(test_unseen)
    decoded3 = tokenizer.decode(ids3)
    print(f"\nTest 3 — Unseen words:")
    print(f"  Input   : {test_unseen}")
    print(f"  Tokens  : {len(ids3)}")
    print(f"  Decoded : {decoded3[:80]}")

    # Test 4 — QA encoding
    sample = train_data[0]
    qa_enc = tokenizer.encode_qa(
        sample['document'],
        sample['question'],
        max_doc_length=512,
        max_q_length=32
    )
    print(f"\nTest 4 — QA encoding:")
    print(f"  Document tokens : {qa_enc['doc_length']}")
    print(f"  Question tokens : {qa_enc['que_length']}")
    print(f"  Total tokens    : {qa_enc['total_length']}")
    print(f"  Segment 0 (doc) : {sum(1 for s in qa_enc['segment_mask'] if s == 0)} tokens")
    print(f"  Segment 1 (que) : {sum(1 for s in qa_enc['segment_mask'] if s == 1)} tokens")

    # Test 5 — Position tracking
    test_pos = "hello world"
    ids5, pos5 = tokenizer.encode(test_pos)
    print(f"\nTest 5 — Position tracking:")
    for tid, p in zip(ids5[:8], pos5[:8]):
        print(f"  token={tokenizer.id2token.get(tid,'?'):12s} id={tid:5d}  char_pos={p}")

    print(f"\nTokenizer ready for PDCL.")
