#!/usr/bin/env python3
"""GPQA-Diamond runner with the simple-evals style prompt (explicit answer
marker instruction), for comparison with the lm_eval cot_zeroshot variant.

  python3 gpqa_simple.py --base-url http://127.0.0.1:8214/v1 --model MODEL \
      --csv /path/gpqa_diamond.csv --seed 1 --out result.json

Same data + shuffle logic as lm_eval's process_docs; prompt follows the
OpenAI simple-evals GPQA template. Sampling is passed through verbatim.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

QUERY_TEMPLATE = (
    "Answer the following multiple choice question. The last line of your "
    "response should be of the following format: 'Answer: $LETTER' (without "
    "quotes) where LETTER is one of {letters}.\n\n"
    "{question}\n\n"
    "A) {choice1}\nB) {choice2}\nC) {choice3}\nD) {choice4}"
)


def preprocess(text):
    if text is None:
        return " "
    return text.strip().replace(" [title]", ". ").replace("  ", " ")


def load(path, seed):
    rng = random.Random(0)  # lm_eval-equivalent base shuffle; seed shifts below
    rows = []
    with open(path, encoding="utf-8") as f:
        for doc in csv.DictReader(f):
            choices = [
                preprocess(doc["Incorrect Answer 1"]),
                preprocess(doc["Incorrect Answer 2"]),
                preprocess(doc["Incorrect Answer 3"]),
                preprocess(doc["Correct Answer"]),
            ]
            rng.shuffle(choices)
            correct = choices.index(preprocess(doc["Correct Answer"]))
            rows.append({
                "question": preprocess(doc["Question"]),
                "choices": choices,
                "answer": "ABCD"[correct],
            })
    return rows


def chat(base_url, model, prompt, temperature, top_p, max_tokens, timeout=7200):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())["choices"][0]["message"]["content"]


def extract(text):
    m = list(re.finditer(r"Answer:\s*\(?([A-D])\)?", text))
    if m:
        return m[-1].group(1)
    m = list(re.finditer(r"[Tt]he answer is[:\s\*]*\(?([A-D])", text))
    if m:
        return m[-1].group(1)
    m = list(re.finditer(r"\(([A-D])\)", text))
    return m[-1].group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=65536)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = load(args.csv, args.seed)
    print(f"[gpqa-simple] {len(rows)} questions, seed={args.seed}, "
          f"T={args.temperature} top_p={args.top_p} max_tokens={args.max_tokens}")

    def one(item):
        i, r = item
        prompt = QUERY_TEMPLATE.format(
            letters="ABCD", question=r["question"],
            choice1=r["choices"][0], choice2=r["choices"][1],
            choice3=r["choices"][2], choice4=r["choices"][3])
        t0 = time.perf_counter()
        resp = chat(args.base_url, args.model, prompt,
                    args.temperature, args.top_p, args.max_tokens)
        pred = extract(resp)
        return {"i": i, "target": r["answer"], "pred": pred,
                "correct": pred == r["answer"],
                "resp_chars": len(resp), "seconds": round(time.perf_counter() - t0, 1),
                "tail": resp[-300:]}

    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        for res in ex.map(one, enumerate(rows)):
            results.append(res)
            if len(results) % 20 == 0:
                print(f"[gpqa-simple] {len(results)}/{len(rows)}", flush=True)

    n = len(results)
    ok = sum(r["correct"] for r in results)
    noext = sum(1 for r in results if r["pred"] is None)
    score = 100.0 * ok / n
    summary = {"score": score, "correct": ok, "n": n, "no_extract": noext,
               "seed": args.seed, "temperature": args.temperature,
               "top_p": args.top_p, "max_tokens": args.max_tokens}
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "samples": results}, f, indent=1)
    print(f"[gpqa-simple] SCORE {score:.2f} ({ok}/{n}), no_extract={noext} -> {args.out}")


if __name__ == "__main__":
    sys.exit(main())
