# grounded-qa-eval-gate

I treat a prompt as code, so it gets a regression test. This repo is an evaluation gate for an LLM extraction prompt. The model reads a passage and a question and returns JSON with an answer and a quote copied from the passage. CI fails when a prompt change makes the scores worse.

## How the gate works

1. `data/golden.jsonl` holds 100 SQuAD 2.0 questions with human labels. 60 have an answer in the passage and 40 do not.
2. `python gate.py record` sends each item to Gemini with the prompt and saves the raw replies in `recordings/<prompt hash>.jsonl`.
3. `python gate.py score <recording>` checks each reply. The JSON must match the schema. The quote must appear word for word in the passage after I collapse whitespace, and the answer must sit inside the quote. I count an answer as correct by SQuAD exact match or token F1 of at least 0.6, and an unanswerable item as correct only when the model declines.
4. `python gate.py check` hashes `prompts/extract.txt`, scores the recording for that hash and compares it with `baseline.json`. It fails when accuracy drops more than 2 points, when the unsupported answer rate rises more than 2 points, or when nobody has recorded the current prompt yet.
5. CI runs the tests and the check on every push and pull request. It scores the committed recordings, so it needs no API key.

## Results

| Prompt | Accuracy | Grounded quote rate | Unsupported answer rate |
|---|---|---|---|
| `prompts/extract.txt` | 90.0% | 100.0% of 62 answered | 4.0% |
| `prompts/extract_no_quote_rule.txt` | 73.0% | 40.8% of 71 answered | 16.0% |

Model `gemini-3.5-flash-lite`, recorded on 14 November 2026. Both recordings are in `recordings/` and `baseline.json` holds the first row.

The second prompt is the same task without the grounding and decline rules. The gate fails it. Accuracy falls 17 points and the unsupported answer rate rises 12 points.

Accuracy and the unsupported answer rate count all 100 items. The grounded quote rate counts only the items the model answered. An unsupported answer is an answer to an unanswerable question or an answer whose quote is missing from the passage.

## Run it

```
pip install pytest
pytest -q
python gate.py check
python gate.py score recordings/<hash>.jsonl
```

## Record again

Put your own key in `GEMINI_API_KEY` and run `python gate.py record`. Add `prompts/extract_no_quote_rule.txt` as an argument to record the weaker prompt. The script waits 7 seconds between calls and backs off on HTTP 429. It resumes a partial file, so delete the file in `recordings/` to start from scratch. `python make_golden.py dev-v2.0.json` rebuilds the golden set with seed 42.

## Data and licence

The passages and questions come from the [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/) dev set by Pranav Rajpurkar, Robin Jia and Percy Liang ([paper](https://arxiv.org/abs/1806.03822), ACL 2018). The authors release it under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). I sampled 100 items without editing them and share `data/golden.jsonl` under the same licence.
