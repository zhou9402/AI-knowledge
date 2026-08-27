#!/usr/bin/env python3
"""AA-LCR (Artificial Analysis Long Context Reasoning) runner for an
OpenAI-compatible endpoint. Stdlib only.

Generation pass:
  python3 aa_lcr.py generate --base-url http://127.0.0.1:8126/v1 \
      --model MODEL --out responses.jsonl [--repeats 1] [--limit N]

Judge pass (needs a separate judge endpoint, e.g. Qwen3-235B non-reasoning):
  python3 aa_lcr.py judge --base-url http://127.0.0.1:8127/v1 \
      --model JUDGE --in responses.jsonl --out judged.jsonl

Dataset: ArtificialAnalysis/AA-LCR (public, Apache-2.0). The zip is fetched
once into --data-dir. Documents are concatenated in data_source_filenames
order per the official prompt template.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import unicodedata
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CSV_URL = "https://huggingface.co/datasets/ArtificialAnalysis/AA-LCR/resolve/main/AA-LCR_Dataset.csv"
ZIP_URL = "https://huggingface.co/datasets/ArtificialAnalysis/AA-LCR/resolve/main/extracted_text/AA-LCR_extracted-text.zip"

PROMPT_TEMPLATE = """BEGIN INPUT DOCUMENTS

{documents_text}

END INPUT DOCUMENTS

Answer the following question using the input documents provided above.

START QUESTION

{question}

END QUESTION
"""

JUDGE_TEMPLATE = """Assess whether the following CANDIDATE ANSWER is CORRECT or INCORRECT.
For the CANDIDATE ANSWER to be correct, it must be consistent with the OFFICIAL ANSWER.

The question, for reference only: {question}
The OFFICIAL ANSWER: {official_answer}
CANDIDATE ANSWER TO ASSESS: {candidate_answer}

Reply only with CORRECT or INCORRECT.
"""


def ensure_dataset(data_dir: Path) -> tuple[Path, Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "AA-LCR_Dataset.csv"
    docs_dir = data_dir / "extracted"
    if not csv_path.exists():
        print(f"[aa-lcr] downloading {CSV_URL}", flush=True)
        urllib.request.urlretrieve(CSV_URL, csv_path)
    if not (docs_dir / "lcr").exists():
        zip_path = data_dir / "AA-LCR_extracted-text.zip"
        if not zip_path.exists():
            print(f"[aa-lcr] downloading {ZIP_URL} (~large)", flush=True)
            urllib.request.urlretrieve(ZIP_URL, zip_path)
        print("[aa-lcr] extracting document text zip", flush=True)
        with zipfile.ZipFile(zip_path) as zf:
            for info in zf.infolist():
                # The zip stores names without the UTF-8 flag; fix the
                # CP437-misdecoded mojibake (e.g. "EUΓÇÖs" -> "EU’s").
                try:
                    fixed = info.filename.encode("cp437").decode("utf-8")
                except (UnicodeDecodeError, UnicodeEncodeError):
                    fixed = info.filename
                # Normalize to NFC: the CSV uses precomposed form while some
                # zip entries extract to NFD (e.g. ş = s + U+0327).
                fixed = unicodedata.normalize("NFC", fixed)
                if info.is_dir():
                    (docs_dir / fixed).mkdir(parents=True, exist_ok=True)
                    continue
                target = docs_dir / fixed
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, open(target, "wb") as dst:
                    dst.write(src.read())
    # Zip contains a top-level lcr/ folder (per official loader comments).
    return csv_path, docs_dir


def load_questions(csv_path: Path) -> list[dict]:
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if isinstance(row.get("data_source_filenames"), str):
            row["data_source_filenames"] = row["data_source_filenames"].split(";")
    return rows


def build_prompt(row: dict, docs_root: Path) -> str:
    # Official loader: documents live under lcr/lcr/<category>/<set_id>/ after
    # extraction; tolerate both nestings.
    base = None
    for cand in (docs_root / "lcr" / "lcr", docs_root / "lcr"):
        if (cand / row["document_category"] / row["document_set_id"]).is_dir():
            base = cand
            break
    if base is None:
        raise FileNotFoundError(
            f"document set not found for {row['document_category']}/{row['document_set_id']} under {docs_root}"
        )
    docs = []
    set_dir = base / row["document_category"] / row["document_set_id"]
    for name in row["data_source_filenames"]:
        path = set_dir / name
        if not path.exists():
            # Defensive: CP437-mojibake or NFC/NFD normalization variants.
            candidates = []
            try:
                candidates.append(name.encode("utf-8").decode("cp437"))
            except (UnicodeDecodeError, UnicodeEncodeError):
                pass
            candidates.append(unicodedata.normalize("NFD", name))
            candidates.append(unicodedata.normalize("NFC", name))
            for alt_name in candidates:
                alt = set_dir / alt_name
                if alt.exists():
                    path = alt
                    break
        docs.append(path.read_text(encoding="utf-8"))
    documents_text = "\n\n".join(
        f"BEGIN DOCUMENT {i + 1}:\n{doc}\nEND DOCUMENT {i + 1}" for i, doc in enumerate(docs)
    )
    return PROMPT_TEMPLATE.format(documents_text=documents_text, question=row["question"])


def chat(base_url: str, model: str, prompt: str, max_tokens: int,
         temperature: float, top_p: float, timeout: float = 7200.0) -> str:
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
        out = json.loads(resp.read())
    return out["choices"][0]["message"]["content"]


def cmd_generate(args: argparse.Namespace) -> None:
    csv_path, docs_root = ensure_dataset(Path(args.data_dir))
    questions = load_questions(csv_path)
    if args.limit:
        questions = questions[: args.limit]
    out_path = Path(args.out)
    done = set()
    if out_path.exists():
        done = {json.loads(l)["qid"] for l in out_path.read_text().splitlines() if l.strip()}

    def one(item):
        idx, row = item
        qid = f"q{idx:03d}"
        if qid in done:
            return None
        prompt = build_prompt(row, docs_root)
        recs = []
        for r in range(args.repeats):
            t0 = time.perf_counter()
            resp = chat(args.base_url, args.model, prompt,
                        args.max_tokens, args.temperature, args.top_p)
            recs.append({"repeat": r, "response": resp,
                         "seconds": round(time.perf_counter() - t0, 2)})
        return {
            "qid": qid,
            "question": row["question"],
            "official_answer": row["answer"],
            "category": row.get("document_category"),
            "prompt_chars": len(prompt),
            "runs": recs,
        }

    with open(out_path, "a", encoding="utf-8") as fout, ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        for res in ex.map(one, enumerate(questions)):
            if res is None:
                continue
            fout.write(json.dumps(res, ensure_ascii=False) + "\n")
            fout.flush()
            print(f"[aa-lcr] {res['qid']} done ({res['prompt_chars']} chars prompt)", flush=True)
    print(f"[aa-lcr] generate done -> {out_path}", flush=True)


def cmd_judge(args: argparse.Namespace) -> None:
    in_path = Path(getattr(args, "in"))
    out_path = Path(args.out)
    n_correct = n_total = 0
    with open(in_path, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            rec = json.loads(line)
            verdicts = []
            for run in rec["runs"]:
                prompt = JUDGE_TEMPLATE.format(
                    question=rec["question"],
                    official_answer=rec["official_answer"],
                    candidate_answer=run["response"],
                )
                v = chat(args.base_url, args.model, prompt, max_tokens=16,
                         temperature=0.0, top_p=1.0).strip().upper()
                # First whitespace/punctuation-delimited token decides;
                # note "INCORRECT" *contains* "CORRECT" — substring tests are wrong.
                first = v.split()[0].strip(".") if v.split() else ""
                verdicts.append(first == "CORRECT")
            rec["verdicts"] = verdicts
            n_correct += sum(verdicts)
            n_total += len(verdicts)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    score = 100.0 * n_correct / max(n_total, 1)
    print(f"[aa-lcr] JUDGED {n_correct}/{n_total} = {score:.2f}  -> {out_path}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("generate", "judge"):
        p = sub.add_parser(name)
        p.add_argument("--base-url", required=True)
        p.add_argument("--model", required=True)
        p.add_argument("--out", required=True)
        if name == "judge":
            p.add_argument("--in", dest="in", required=True)
        else:
            p.add_argument("--data-dir", default=str(Path(__file__).parent / "extern" / "aa-lcr"))
            p.add_argument("--repeats", type=int, default=1, help="AA runs 3; default 1")
            p.add_argument("--limit", type=int, default=0)
            p.add_argument("--concurrency", type=int, default=4)
            p.add_argument("--max-tokens", type=int, default=32768)
            p.add_argument("--temperature", type=float, default=1.0)
            p.add_argument("--top-p", type=float, default=0.95)
    args = ap.parse_args()
    (cmd_generate if args.cmd == "generate" else cmd_judge)(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
