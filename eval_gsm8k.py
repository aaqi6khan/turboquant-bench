#!/usr/bin/env python3
"""GSM8K exact-match eval against local vLLM OpenAI endpoint."""
import json, re, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor

TAG = sys.argv[1]
BASE = "http://localhost:8001/v1/chat/completions"
import os
SC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
rows = json.load(open(f"{SC}/gsm8k_100.json"))

def ask(row):
    body = json.dumps({
        "model": "qwen3-8b",
        "messages": [{"role": "user", "content": row["q"] + "\n\nSolve step by step, then give the final numeric answer on the last line as: #### <number>"}],
        "max_tokens": 768,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(BASE, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            text = json.load(r)["choices"][0]["message"]["content"]
    except Exception as e:
        return (row, "", f"ERROR {e}")
    m = re.findall(r"####\s*\$?(-?[\d,]+(?:\.\d+)?)", text)
    pred = m[-1].replace(",", "") if m else ""
    return (row, pred, text[-200:])

with ThreadPoolExecutor(max_workers=8) as ex:
    results = list(ex.map(ask, rows))

correct = 0
wrong = []
for row, pred, tail in results:
    gold = row["a"].replace(",", "")
    ok = pred.rstrip("0").rstrip(".") == gold.rstrip("0").rstrip(".") if "." in pred else pred == gold
    correct += ok
    if not ok:
        wrong.append({"q": row["q"][:80], "gold": gold, "pred": pred, "tail": tail})

print(f"{TAG}: {correct}/100 correct")
json.dump({"tag": TAG, "correct": correct, "wrong": wrong}, open(f"{SC}/gsm8k_{TAG}.json", "w"), indent=1)
