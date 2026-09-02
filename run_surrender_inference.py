#!/usr/bin/env python3
"""One-token yes/no inference over the surrender prompts, against a vLLM OpenAI endpoint.

Sends each prompt with max_tokens=1 and records the top-k logprobs, so we keep the
continuous P(yes) signal instead of only the argmax. Writes a resumable JSONL and a CSV.

Typical use (tunnel first: `ssh -N harold-llm`):
    python run_surrender_inference.py --model ministral-8b
    python run_surrender_inference.py --model ministral-8b --limit 20     # smoke test
    python run_surrender_inference.py --dry-run                           # no server needed
"""

import argparse
import json
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

DATASET = "sycophancy_prompt_dataset_gemma_no-sampling_v3.json"
OUT_DIR = "responses"
BASE_URL = "http://localhost:8080/v1"  # harold-llm forwards this to remote 18080
QUESTION = "surrender"
META_FIELDS = ["crime", "severity", "moral_stance", "circumstance", "victim_gender",
               "scenario_id", "relation", "deixis", "question"]

# Surface forms the tokenizer may emit. "Ġ"/"▁" are the BPE/SentencePiece space markers.
YES_FORMS = {"yes", "y"}
NO_FORMS = {"no", "n"}


def normalize_token(token):
    return token.replace("▁", "").replace("Ġ", "").strip().strip("*_\"'").lower()


def classify(token):
    n = normalize_token(token)
    if n in YES_FORMS:
        return "yes"
    if n in NO_FORMS:
        return "no"
    return None


def score_top_logprobs(entries):
    """Sum probability mass over every surface form of yes and of no."""
    p = {"yes": 0.0, "no": 0.0}
    for e in entries:
        label = classify(e["token"])
        if label:
            p[label] += math.exp(e["logprob"])
    mass = p["yes"] + p["no"]
    return {
        "p_yes": p["yes"],
        "p_no": p["no"],
        "p_yesno_mass": mass,
        "p_yes_norm": p["yes"] / mass if mass > 0 else None,
        "logprob_yes": math.log(p["yes"]) if p["yes"] > 0 else None,
        "logprob_no": math.log(p["no"]) if p["no"] > 0 else None,
    }


def build_payload(prompt, args):
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "seed": args.seed,
        "logprobs": True,
        "top_logprobs": args.top_logprobs,
    }
    if args.disable_thinking:
        # Qwen3 opens every reply with <think>; that token would be the only one we read.
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    return payload


def call_one(session, row, args):
    payload = build_payload(row["prompt"], args)
    url = f"{args.base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}

    last_error = None
    for attempt in range(args.retries):
        try:
            r = session.post(url, json=payload, headers=headers, timeout=args.timeout)
            r.raise_for_status()
            choice = r.json()["choices"][0]
            content = choice["logprobs"]["content"][0]
            out = {
                "row_id": row["row_id"],
                "model": args.model,
                "top1_token": content["token"],
                "top1_logprob": content["logprob"],
                "text": choice["message"].get("content"),
                "top_logprobs": [
                    {"token": e["token"], "logprob": e["logprob"]}
                    for e in content["top_logprobs"]
                ],
                "error": None,
            }
            out.update(score_top_logprobs(out["top_logprobs"]))
            out["answer"] = classify(content["token"]) or "other"
            return out
        except Exception as exc:  # noqa: BLE001 - retry on anything transient
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < args.retries - 1:
                time.sleep(2 ** attempt)

    return {"row_id": row["row_id"], "model": args.model, "error": last_error,
            "answer": None, "p_yes": None, "p_no": None, "p_yes_norm": None,
            "p_yesno_mass": None, "logprob_yes": None, "logprob_no": None,
            "top1_token": None, "top1_logprob": None, "top_logprobs": None, "text": None}


def load_rows(args):
    data = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    rows = [{**r, "row_id": i} for i, r in enumerate(data) if r["question"] == args.question]
    if not args.limit:
        return rows

    # Subsample whole scenarios, evenly spaced: every relation x deixis cell stays
    # complete, so a pilot supports the same paired contrasts as the full run.
    scenarios = sorted({r["scenario_id"] for r in rows})
    per_scenario = len(rows) / len(scenarios)
    n = min(len(scenarios), max(1, round(args.limit / per_scenario)))
    step = len(scenarios) / n
    keep = {scenarios[int(i * step)] for i in range(n)}
    return [r for r in rows if r["scenario_id"] in keep]


def already_done(path):
    if not Path(path).exists():
        return set()
    done = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            if rec.get("error") is None:
                done.add(rec["row_id"])
    return done


def run(rows, args):
    lock = threading.Lock()
    out_file = open(args.jsonl, "a", encoding="utf-8")
    session = requests.Session()
    session.mount("http://", requests.adapters.HTTPAdapter(pool_maxsize=args.concurrency))
    counter = {"n": 0}

    def work(row):
        result = call_one(session, row, args)
        with lock:
            out_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_file.flush()
            counter["n"] += 1
            if counter["n"] % 50 == 0 or counter["n"] == len(rows):
                print(f"  {counter['n']}/{len(rows)}", file=sys.stderr, flush=True)

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        list(pool.map(work, rows))
    out_file.close()
    session.close()


def write_csv(args):
    import pandas as pd

    data = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    meta = {i: r for i, r in enumerate(data)}

    records = []
    for line in Path(args.jsonl).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        src = meta[rec["row_id"]]
        row = {"row_id": rec["row_id"]}
        row.update({k: src.get(k) for k in META_FIELDS})
        row.update({k: rec.get(k) for k in
                    ["model", "answer", "top1_token", "top1_logprob", "p_yes", "p_no",
                     "p_yes_norm", "p_yesno_mass", "logprob_yes", "logprob_no", "text",
                     "error"]})
        row["top_logprobs_json"] = json.dumps(rec.get("top_logprobs"), ensure_ascii=False)
        records.append(row)

    df = pd.DataFrame(records).drop_duplicates(subset=["row_id", "model"], keep="last")
    df = df.sort_values(["model", "row_id"])
    df.to_csv(args.out, index=False)
    return df


def report(df):
    ok = df[df["error"].isna()]
    print(f"\nrows: {len(df)}  ok: {len(ok)}  errors: {len(df) - len(ok)}")
    if ok.empty:
        return
    print("\nfirst-token answer distribution:")
    print(ok["answer"].value_counts().to_string())
    mass = ok["p_yesno_mass"].dropna()
    if len(mass):
        print(f"\nyes/no probability mass  median={mass.median():.3f}  "
              f"p10={mass.quantile(0.10):.3f}  frac<0.5: {(mass < 0.5).mean():.1%}")
        print("  (low mass = model is not answering in the expected one-token format)")
    # Refusal that concentrates on particular crimes makes the missingness
    # non-random, which biases the within-scenario proximity contrast.
    ok = ok.copy()
    ok["off_format"] = ok["answer"].ne("yes") & ok["answer"].ne("no")
    by_crime = ok.groupby(ok["crime"].str.slice(0, 40)).agg(
        n=("off_format", "size"), off_format=("off_format", "mean"),
        mean_mass=("p_yesno_mass", "mean")).sort_values("off_format", ascending=False)
    print("\noff-format / refusal rate by crime:")
    print(by_crime.round(3).to_string())

    print("\nmean p_yes_norm by deixis x relation:")
    print(ok.pivot_table(index="relation", columns="deixis",
                         values="p_yes_norm", aggfunc="mean").round(3).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--question", default=QUESTION,
                    choices=["surrender", "severity", "severity_judge"])
    ap.add_argument("--model", default=os.environ.get("VLLM_MODEL", ""),
                    help="model id as served by vLLM (see GET /v1/models)")
    ap.add_argument("--base-url", default=os.environ.get("VLLM_BASE_URL", BASE_URL))
    ap.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", ""))
    ap.add_argument("--out", default=None, help="CSV path (default: results_<model>.csv)")
    ap.add_argument("--jsonl", default=None, help="raw JSONL path, used for resume")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--top-logprobs", type=int, default=20)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--disable-thinking", action="store_true",
                    help="Qwen3 only (not Qwen2.5): suppress the <think> prefix")
    ap.add_argument("--max-tokens", type=int, default=1,
                    help="raise (e.g. 64) to also capture refusal text; the first-token "
                         "logprobs are unchanged because decoding is greedy")
    ap.add_argument("--limit", type=int, default=0,
                    help="evenly-spaced subsample across all cells, for smoke tests")
    ap.add_argument("--fresh", action="store_true", help="ignore existing JSONL, start over")
    ap.add_argument("--dry-run", action="store_true",
                    help="print one request body and exit; no server needed")
    ap.add_argument("--csv-only", action="store_true",
                    help="rebuild the CSV from an existing JSONL without querying")
    args = ap.parse_args()

    tag = (args.model or "model").replace("/", "_")
    Path(OUT_DIR).mkdir(exist_ok=True)
    args.out = args.out or str(Path(OUT_DIR) / f"results_{args.question}_{tag}.csv")
    args.jsonl = args.jsonl or str(Path(OUT_DIR) / f"raw_{args.question}_{tag}.jsonl")

    rows = load_rows(args)
    print(f"{len(rows)} prompts for question={args.question!r}", file=sys.stderr)

    if args.dry_run:
        print(json.dumps(build_payload(rows[0]["prompt"], args), indent=2, ensure_ascii=False))
        print(f"\n--> POST {args.base_url.rstrip('/')}/chat/completions", file=sys.stderr)
        print(f"--> would write {args.jsonl} and {args.out}", file=sys.stderr)
        return

    if not args.csv_only:
        if not args.model:
            ap.error("--model is required (query GET /v1/models to see what is served)")
        if args.fresh and Path(args.jsonl).exists():
            Path(args.jsonl).rename(args.jsonl + ".bak")
        done = already_done(args.jsonl)
        todo = [r for r in rows if r["row_id"] not in done]
        print(f"{len(done)} already done, {len(todo)} to go", file=sys.stderr)
        if todo:
            run(todo, args)

    report(write_csv(args))
    print(f"\nwrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
