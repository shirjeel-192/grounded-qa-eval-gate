"""Sample the golden set from the SQuAD 2.0 dev set (CC BY-SA 4.0).

Usage: python make_golden.py dev-v2.0.json
Source: https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json
"""
import json
import random
import sys

SEED = 42
N_ANSWERABLE, N_UNANSWERABLE, MAX_CHARS = 60, 40, 1200


def main(path):
    answerable, unanswerable = [], []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)["data"]
    for article in data:
        for para in article["paragraphs"]:
            if len(para["context"]) >= MAX_CHARS:
                continue
            for qa in para["qas"]:
                item = {
                    "id": qa["id"],
                    "article": article["title"],
                    "passage": para["context"],
                    "question": qa["question"],
                    "gold_answers": sorted({a["text"] for a in qa["answers"]}),
                    "answerable": not qa["is_impossible"],
                }
                (answerable if item["answerable"] else unanswerable).append(item)
    rng = random.Random(SEED)
    items = rng.sample(answerable, N_ANSWERABLE) + rng.sample(unanswerable, N_UNANSWERABLE)
    articles = {item["article"] for item in items}
    assert len(articles) >= 15, f"only {len(articles)} articles, need 15"
    with open("data/golden.jsonl", "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"wrote {len(items)} items from {len(articles)} articles")


if __name__ == "__main__":
    main(sys.argv[1])
