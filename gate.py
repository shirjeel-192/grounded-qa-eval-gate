"""Evaluation gate for a grounded extraction prompt.

python gate.py score <recording>   print the three numbers for a recording
python gate.py check               score the current prompt's recording against baseline.json
python gate.py record [prompt]     record model responses with your own GEMINI_API_KEY
"""
import datetime
import hashlib
import json
import os
import re
import string
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent
PROMPT = ROOT / "prompts" / "extract.txt"
GOLDEN = ROOT / "data" / "golden.jsonl"
BASELINE = ROOT / "baseline.json"
MODEL = "gemini-3.5-flash-lite"
TOLERANCE = 2.0  # percentage points
F1_THRESHOLD = 0.6
SECONDS_BETWEEN_CALLS = 7  # keeps us under 10 requests a minute on the free tier
KEYS = {"answerable": bool, "answer": str, "quote": str}


# Scoring

def squash(text):
    """Collapse runs of whitespace. This is the only change allowed before the verbatim check."""
    return " ".join(text.split())


def normalize_answer(text):
    """SQuAD normalisation. Lower case, drop punctuation and articles, collapse whitespace."""
    text = "".join(ch for ch in text.lower() if ch not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def token_f1(prediction, gold):
    pred, ref = normalize_answer(prediction).split(), normalize_answer(gold).split()
    common = sum((Counter(pred) & Counter(ref)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(pred), common / len(ref)
    return 2 * precision * recall / (precision + recall)


def parse(raw):
    """Return the response as a dict if it is strict JSON with exactly the right keys and types."""
    try:
        obj = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(obj, dict) or set(obj) != set(KEYS):
        return None
    if not all(isinstance(obj[key], kind) for key, kind in KEYS.items()):
        return None
    return obj


def score_item(item, raw):
    obj = parse(raw)
    valid = obj is not None
    answered = valid and obj["answerable"]
    quote = squash(obj["quote"]) if valid else ""
    answer = squash(obj["answer"]) if valid else ""
    quote_in_passage = bool(quote) and quote in squash(item["passage"])
    grounded = answered and quote_in_passage and bool(answer) and answer in quote
    if item["answerable"]:
        correct = answered and any(
            normalize_answer(answer) == normalize_answer(gold) or token_f1(answer, gold) >= F1_THRESHOLD
            for gold in item["gold_answers"]
        )
    else:
        correct = valid and not obj["answerable"]
    unsupported = answered and (not item["answerable"] or not quote_in_passage)
    return {"valid": valid, "answered": answered, "grounded": grounded,
            "correct": correct, "unsupported": unsupported}


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize(responses):
    """Score every golden item. A missing response counts as malformed."""
    rows = [score_item(item, responses.get(item["id"])) for item in load_jsonl(GOLDEN)]
    answered = [row for row in rows if row["answered"]]

    def pct(part, whole):
        return round(100 * part / whole, 1) if whole else 0.0

    return {
        "accuracy": pct(sum(r["correct"] for r in rows), len(rows)),
        "grounded_quote_rate": pct(sum(r["grounded"] for r in answered), len(answered)),
        "unsupported_answer_rate": pct(sum(r["unsupported"] for r in rows), len(rows)),
        "items": len(rows),
        "answered": len(answered),
        "schema_valid": sum(r["valid"] for r in rows),
    }


# Recordings

def prompt_hash(prompt_path):
    return hashlib.sha256(Path(prompt_path).read_bytes()).hexdigest()[:12]


def recording_path(prompt_path):
    return ROOT / "recordings" / f"{prompt_hash(prompt_path)}.jsonl"


def load_recording(path):
    """First line is a header with the model and date. Every other line is one response."""
    header, *rows = load_jsonl(path)
    return header, {row["id"]: row["raw"] for row in rows}


def print_scores(path, header, scores):
    print(f"recording                {Path(path).name} ({header['model']}, {header['date']})")
    print(f"accuracy                 {scores['accuracy']:.1f}%")
    print(f"grounded quote rate      {scores['grounded_quote_rate']:.1f}% of {scores['answered']} answered")
    print(f"unsupported answer rate  {scores['unsupported_answer_rate']:.1f}%")
    print(f"schema valid             {scores['schema_valid']}/{scores['items']}")


def score(path):
    header, responses = load_recording(path)
    print_scores(path, header, summarize(responses))
    return 0


def regressions(scores, baseline):
    problems = []
    if round(baseline["accuracy"] - scores["accuracy"], 1) > TOLERANCE:
        problems.append(f"accuracy fell from {baseline['accuracy']}% to {scores['accuracy']}%")
    rise = round(scores["unsupported_answer_rate"] - baseline["unsupported_answer_rate"], 1)
    if rise > TOLERANCE:
        problems.append(f"unsupported answer rate rose from {baseline['unsupported_answer_rate']}%"
                        f" to {scores['unsupported_answer_rate']}%")
    return problems


def check():
    path = recording_path(PROMPT)
    if not path.exists():
        print(f"FAIL no recording for prompt hash {prompt_hash(PROMPT)}. "
              f"Record responses for the changed prompt with: python gate.py record")
        return 1
    header, responses = load_recording(path)
    scores = summarize(responses)
    baseline = json.loads(BASELINE.read_text())
    print_scores(path, header, scores)
    problems = regressions(scores, baseline)
    for problem in problems:
        print(f"FAIL {problem} (tolerance {TOLERANCE} points)")
    if not problems:
        print(f"PASS within {TOLERANCE} points of baseline.json")
    return 1 if problems else 0


# Recording with the Gemini REST API

def call_gemini(text, key):
    body = {"contents": [{"role": "user", "parts": [{"text": text}]}],
            "generationConfig": {"responseMimeType": "application/json",
                                 "thinkingConfig": {"thinkingLevel": "low"}}}
    request = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "x-goog-api-key": key},
    )
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                data = json.load(response)
            parts = data["candidates"][0].get("content", {}).get("parts", [])
            return "".join(part.get("text", "") for part in parts), data.get("modelVersion")
        except urllib.error.HTTPError as err:
            detail = err.read().decode(errors="replace")
            if err.code not in (429, 500, 503) or attempt == 4:
                raise SystemExit(f"HTTP {err.code} from Gemini, stopping.\n{detail[:800]}")
            hint = re.search(r'"retryDelay":\s*"(\d+)', detail)
            wait = int(hint.group(1)) + 5 if hint else 30 * 2 ** attempt
            if wait > 600:  # a daily quota, so stop now and resume later
                raise SystemExit(f"Quota used up, retry after {wait}s. Run record again to resume.\n"
                                 f"{detail[:800]}")
            print(f"HTTP {err.code}, waiting {wait}s before retry {attempt + 1}", file=sys.stderr)
            time.sleep(wait)
        except (TimeoutError, urllib.error.URLError) as err:
            if attempt == 4:
                raise SystemExit(f"Gemini did not respond ({err}), stopping.")
            print(f"{err}, retrying in 60s", file=sys.stderr)
            time.sleep(60)


def record(prompt_path=PROMPT):
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit("Set GEMINI_API_KEY to record responses.")
    template = Path(prompt_path).read_text(encoding="utf-8")
    path = recording_path(prompt_path)
    done = set(load_recording(path)[1]) if path.exists() else set()
    if not path.exists():
        header = {"model": MODEL, "date": datetime.date.today().isoformat(),
                  "prompt": Path(os.path.relpath(prompt_path, ROOT)).as_posix(),
                  "prompt_hash": prompt_hash(prompt_path)}
        path.write_text(json.dumps(header) + "\n", encoding="utf-8")
    todo = [item for item in load_jsonl(GOLDEN) if item["id"] not in done]
    print(f"recording {len(todo)} responses into {path.name}", file=sys.stderr)
    with path.open("a", encoding="utf-8") as out:
        for n, item in enumerate(todo, 1):
            text = template.replace("{passage}", item["passage"]).replace("{question}", item["question"])
            raw, version = call_gemini(text, key)
            row = {"id": item["id"], "raw": raw, "model_version": version}
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            print(f"{n}/{len(todo)}", file=sys.stderr)
            time.sleep(SECONDS_BETWEEN_CALLS)
    return 0


if __name__ == "__main__":
    commands = {"score": score, "check": check, "record": record}
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        raise SystemExit(__doc__)
    sys.exit(commands[sys.argv[1]](*sys.argv[2:]))
