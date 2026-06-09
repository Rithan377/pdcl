"""Full dataset alignment check."""
import json
from pdcl_tokenizer import BPETokenizer
from pdcl_dimension_engine import align_answer_to_tokens, AnswerAlignmentError

tokenizer = BPETokenizer()
tokenizer.load('./tokenizer')

for split in ['train', 'val']:
    with open(f'./data/{split}.json') as f:
        data = json.load(f)
    
    correct = 0
    wrong = 0
    skipped = 0
    for sample in data:
        try:
            qa_enc, ans_s, ans_e = align_answer_to_tokens(sample, tokenizer, 384, 32)
            token_ids = qa_enc['token_ids']
            span_ids = token_ids[ans_s: ans_e + 1]
            decoded = tokenizer.decode(span_ids).replace(" ", "").lower()
            answer = sample['answer'].strip().replace(" ", "").lower()
            if answer in decoded or decoded in answer:
                correct += 1
            else:
                wrong += 1
        except AnswerAlignmentError:
            skipped += 1

    total = len(data)
    print(f"{split}: {correct} correct, {wrong} wrong, {skipped} skipped out of {total} ({correct/total*100:.1f}% usable)")
