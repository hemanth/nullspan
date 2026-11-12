#!/usr/bin/env python3
"""Generate synthetic in-distribution Typed Decisions cases by recombining latent factors.

Interfaces (`plan.md` Task 4):
  python training/synth_td_cases.py --n 6000 --seed 7 --out data/td_synth/cases.jsonl

Design guarantees:
- Extracts factor vocabularies from `extract_factor_vocabs(ds_train)` plus empirical
  per-workflow factor value pools from `LocalLLaMA/typed-decisions` `train`.
- Reuses the exact `train` question schema for each workflow.
- Authors a fresh JSON `state` per recombined factor tuple with an LLM conditioned on
  the workflow's state schema and two randomly chosen `train` state exemplars.
- Enforces a strict 12-token shingle leak guard against `test` states, reading ONLY
  the `state` column of `split="test"` and NEVER touching any label/target column.
"""
from __future__ import annotations

import argparse
import collections
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import time
from typing import Any
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.train_systemone import extract_factor_vocabs

TOKEN_RE = re.compile(r"[a-z0-9_]+")

WORKFLOW_REQUIRED_TOP_KEYS: dict[str, set[str]] = {
    "agent_trace_observability": {"agent", "constraints", "task", "trace_summary"},
    "customer_service": {"account", "thread"},
    "invoice_processing": {"delivery", "invoice", "payment", "purchase_order", "vendor_history"},
    "security_incidents": {"alert", "context", "history", "principal"},
}


def tokenize_for_shingles(state_obj: Any) -> list[str]:
    """Normalize a state JSON/string into lowercase alphanumeric tokens."""
    if not isinstance(state_obj, str):
        text = json.dumps(state_obj, ensure_ascii=False, sort_keys=True)
    else:
        text = state_obj
    return TOKEN_RE.findall(text.lower())


def build_shingle_set(states: list[Any], k: int = 12) -> set[tuple[str, ...]]:
    """Build all contiguous k-token shingles across a collection of states."""
    shingles: set[tuple[str, ...]] = set()
    for st in states:
        toks = tokenize_for_shingles(st)
        if len(toks) < k:
            if toks:
                shingles.add(tuple(toks))
            continue
        for i in range(len(toks) - k + 1):
            shingles.add(tuple(toks[i : i + k]))
    return shingles


def has_shingle_overlap(
    candidate_state: Any,
    forbidden_shingles: set[tuple[str, ...]],
    k: int = 12,
) -> bool:
    """Return True iff `candidate_state` shares any k-token shingle with `forbidden_shingles`."""
    if not forbidden_shingles:
        return False
    toks = tokenize_for_shingles(candidate_state)
    if len(toks) < k:
        return tuple(toks) in forbidden_shingles if toks else False
    for i in range(len(toks) - k + 1):
        if tuple(toks[i : i + k]) in forbidden_shingles:
            return True
    return False


def load_test_state_shingles(k: int = 12) -> tuple[set[tuple[str, ...]], int]:
    """Load `LocalLLaMA/typed-decisions` `test` states ONLY (never answers) for leak rejection."""
    from datasets import load_dataset

    ds_test = load_dataset("LocalLLaMA/typed-decisions", "all", split="test")
    only_states = [row["state"] for row in ds_test]
    return build_shingle_set(only_states, k=k), len(only_states)


def build_workflow_factor_pools(ds_train: Any) -> dict[str, dict[str, Any]]:
    """Extract per-workflow factor value distributions, exact question schemas, and exemplars."""
    discrete_vocabs = extract_factor_vocabs(ds_train)
    by_wf: dict[str, list[Any]] = collections.defaultdict(list)
    for row in ds_train:
        by_wf[str(row["workflow"])].append(row)

    pools: dict[str, dict[str, Any]] = {}
    for wf, rows in sorted(by_wf.items()):
        factor_values: dict[str, list[Any]] = collections.defaultdict(list)
        existing_tuples: set[str] = set()
        existing_states: set[str] = set()
        exemplars: list[dict[str, Any]] = []
        q_schema = json.loads(rows[0]["questions"]) if isinstance(rows[0]["questions"], str) else rows[0]["questions"]

        for row in rows:
            fc = json.loads(row["factors"]) if isinstance(row["factors"], str) else dict(row["factors"])
            st = json.loads(row["state"]) if isinstance(row["state"], str) else row["state"]
            existing_tuples.add(json.dumps(fc, sort_keys=True))
            existing_states.add(json.dumps(st, sort_keys=True))
            exemplars.append({"factors": fc, "state": st})
            for k, v in fc.items():
                factor_values[k].append(v)

        pools[wf] = {
            "workflow": wf,
            "questions": q_schema,
            "discrete_vocabs": discrete_vocabs,
            "factor_values": dict(factor_values),
            "existing_tuples": existing_tuples,
            "existing_states": existing_states,
            "exemplars": exemplars,
        }
    return pools


def sample_recombined_factor_tuple(wf_pool: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """Independently recombine latent factor values for a workflow while keeping arithmetic consistent."""
    wf = wf_pool["workflow"]
    fvals: dict[str, list[Any]] = wf_pool["factor_values"]
    dvocabs: dict[str, list[str]] = wf_pool["discrete_vocabs"]
    existing: set[str] = wf_pool["existing_tuples"]

    for _attempt in range(30):
        out: dict[str, Any] = {}
        for fk, emp_list in sorted(fvals.items()):
            non_null = [v for v in emp_list if v is not None]
            # Use `extract_factor_vocabs` discrete strings where applicable and non-boolean
            if fk in dvocabs and non_null and isinstance(non_null[0], str):
                out[fk] = rng.choice(emp_list)
            else:
                val = rng.choice(emp_list)
                if isinstance(val, list):
                    # Recombine constraint subsets across distinct train cases
                    flat_items = sorted({str(x) for sub in non_null if isinstance(sub, list) for x in sub})
                    k_take = max(1, min(len(flat_items), len(val)))
                    out[fk] = rng.sample(flat_items, k=k_take) if flat_items else list(val)
                elif isinstance(val, int) and not isinstance(val, bool):
                    # Slight jitter within observed empirical integer range for numeric richness
                    ints = [int(x) for x in non_null if isinstance(x, int) and not isinstance(x, bool)]
                    lo, hi = min(ints), max(ints)
                    delta = rng.choice([-2, -1, 0, 0, 1, 2])
                    out[fk] = max(lo, min(hi, int(val) + delta))
                else:
                    out[fk] = val

        # Workflow structural invariants observed in `LocalLLaMA/typed-decisions`
        if wf == "customer_service":
            if not out.get("money_involved"):
                out["amount_usd"] = None
            elif out.get("amount_usd") is None:
                pos_amts = [int(x) for x in fvals.get("amount_usd", []) if isinstance(x, (int, float)) and x]
                out["amount_usd"] = rng.choice(pos_amts) if pos_amts else 99
        elif wf == "invoice_processing":
            days = int(out.get("days_until_due", 10))
            out["overdue"] = bool(days < 0)
            po_tot = round(float(out.get("po_total_usd", 1000.0)) * rng.uniform(0.85, 1.18), 2)
            issue = str(out.get("issue", "none"))
            if issue in ("none", "duplicate_submission", "early_discount_expiring"):
                var_pct = 0.0
            else:
                raw_vp = float(out.get("variance_pct", 5.0))
                var_pct = round(raw_vp if abs(raw_vp) > 0.1 else rng.uniform(2.5, 18.0), 2)
            var_usd = round(po_tot * (var_pct / 100.0), 2)
            inv_tot = round(po_tot + var_usd, 2)
            out["po_total_usd"] = po_tot
            out["variance_pct"] = var_pct
            out["variance_usd"] = var_usd
            out["invoice_total_usd"] = inv_tot

        key_s = json.dumps(out, sort_keys=True)
        if key_s not in existing:
            return out
    return out


def _validate_state_shape(workflow: str, state_obj: Any) -> bool:
    if not isinstance(state_obj, dict):
        return False
    req = WORKFLOW_REQUIRED_TOP_KEYS.get(workflow, set())
    if not req.issubset(set(state_obj.keys())):
        return False
    s_txt = json.dumps(state_obj)
    return 160 <= len(s_txt) <= 4500


def _call_anthropic_batch_states(
    workflow: str,
    factor_batch: list[dict[str, Any]],
    exemplars: list[dict[str, Any]],
    model: str,
    api_key: str,
    temperature: float = 0.85,
) -> list[dict[str, Any]]:
    ex_text = "\n\n".join(
        f"Example {i+1} factors:\n{json.dumps(ex['factors'], ensure_ascii=False)}\n"
        f"Example {i+1} state:\n{json.dumps(ex['state'], ensure_ascii=False)}"
        for i, ex in enumerate(exemplars[:2])
    )
    req_items = "\n".join(
        f"Case {idx}: {json.dumps(fc, ensure_ascii=False)}" for idx, fc in enumerate(factor_batch)
    )
    prompt = (
        f"You are generating realistic synthetic `{workflow}` workflow state JSON objects "
        f"from latent factor tuples.\n\n"
        f"Follow the exact JSON keys, nesting, numeric realism, and concise prose style of these "
        f"two reference examples, while inventing completely fresh names, IDs, dates, IPs, SKUs, "
        f"vendors, emails, and wording (never copy sentences from the examples):\n\n"
        f"{ex_text}\n\n"
        f"Now write one JSON `state` object for each of the {len(factor_batch)} factor tuples below. "
        f"Make sure every factor in the tuple is faithfully reflected in the state (without adding "
        f"extra top-level keys outside the workflow's schema).\n\n"
        f"{req_items}"
    )
    body = {
        "model": model,
        "max_tokens": 6400,
        "temperature": temperature,
        "tools": [
            {
                "name": "emit_states",
                "description": "Emit generated workflow state JSON objects in exact input order.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "states": {
                            "type": "array",
                            "items": {"type": "object"},
                        }
                    },
                    "required": ["states"],
                },
            }
        ],
        "tool_choice": {"type": "tool", "name": "emit_states"},
        "messages": [{"role": "user", "content": prompt}],
    }
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "content-type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            for block in payload.get("content", []):
                if block.get("type") == "tool_use" and block.get("name") == "emit_states":
                    arr = block.get("input", {}).get("states", [])
                    if isinstance(arr, list):
                        return arr
            return []
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503, 529) and attempt < 3:
                time.sleep(2.0 * (attempt + 1))
                continue
            raise
        except Exception:
            if attempt < 3:
                time.sleep(2.0 * (attempt + 1))
                continue
            raise
    return []


def generate_synthetic_cases(
    n_total: int,
    seed: int,
    out_path: Path,
    author_model: str = "claude-haiku-4-5",
    batch_size: int = 3,
    workers: int = 8,
    shingle_k: int = 12,
) -> dict[str, Any]:
    from datasets import load_dataset

    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is required to author synthetic states with Anthropic.")

    ds_train = load_dataset("LocalLLaMA/typed-decisions", "all", split="train")
    pools = build_workflow_factor_pools(ds_train)
    forbidden_shingles, n_test_states = load_test_state_shingles(k=shingle_k)
    print(
        f"Built {len(forbidden_shingles):,} {shingle_k}-token leak-guard shingles from {n_test_states} test states.",
        flush=True,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing_rows: list[dict[str, Any]] = []
    seen_states: set[str] = set()
    for wf_pool in pools.values():
        seen_states.update(wf_pool["existing_states"])

    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                existing_rows.append(r)
                st_obj = json.loads(r["state"]) if isinstance(r["state"], str) else r["state"]
                seen_states.add(json.dumps(st_obj, sort_keys=True))
        print(f"Resuming from {len(existing_rows):,} existing synthetic cases in {out_path}", flush=True)

    workflows = sorted(pools.keys())
    per_wf_target = {wf: n_total // len(workflows) for wf in workflows}
    for i in range(n_total % len(workflows)):
        per_wf_target[workflows[i]] += 1

    counts_by_wf = collections.Counter(str(r["workflow"]) for r in existing_rows)
    rejected_shingle = 0
    rejected_shape = 0
    rng = random.Random(seed + len(existing_rows) * 31)

    with out_path.open("a", encoding="utf-8") as out_fh:
        for wf in workflows:
            wf_pool = pools[wf]
            q_json = json.dumps(wf_pool["questions"], ensure_ascii=False)
            while counts_by_wf[wf] < per_wf_target[wf]:
                rem = per_wf_target[wf] - counts_by_wf[wf]
                # Oversample slightly per round to absorb any shingle/shape rejections cleanly
                n_jobs = max(1, min(workers * 2, (rem + batch_size - 1) // batch_size))
                job_specs = []
                for _ in range(n_jobs):
                    cur_bsz = min(batch_size, max(2, rem))
                    f_batch = [sample_recombined_factor_tuple(wf_pool, rng) for _ in range(cur_bsz)]
                    exs = rng.sample(wf_pool["exemplars"], k=min(2, len(wf_pool["exemplars"])))
                    job_specs.append((f_batch, exs))

                with ThreadPoolExecutor(max_workers=workers) as ex_pool:
                    futs = {
                        ex_pool.submit(
                            _call_anthropic_batch_states,
                            wf,
                            f_batch,
                            exs,
                            author_model,
                            api_key,
                        ): f_batch
                        for f_batch, exs in job_specs
                    }
                    for fut in as_completed(futs):
                        f_batch = futs[fut]
                        states_out = fut.result()
                        for fc, st_cand in zip(f_batch, states_out):
                            if counts_by_wf[wf] >= per_wf_target[wf]:
                                break
                            if not _validate_state_shape(wf, st_cand):
                                rejected_shape += 1
                                continue
                            if has_shingle_overlap(st_cand, forbidden_shingles, k=shingle_k):
                                rejected_shingle += 1
                                continue
                            canon_st = json.dumps(st_cand, sort_keys=True, ensure_ascii=False)
                            if canon_st in seen_states:
                                rejected_shape += 1
                                continue
                            seen_states.add(canon_st)
                            idx_num = sum(counts_by_wf.values())
                            rec = {
                                "id": f"synth_{wf}_{seed}_{idx_num:05d}",
                                "workflow": wf,
                                "split": "synth_train",
                                "state": json.dumps(st_cand, ensure_ascii=False),
                                "questions": q_json,
                                "factors": json.dumps(fc, ensure_ascii=False),
                            }
                            out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            counts_by_wf[wf] += 1
                out_fh.flush()
                print(
                    f"  [{wf}] {counts_by_wf[wf]}/{per_wf_target[wf]} "
                    f"(rejected_shingle={rejected_shingle}, rejected_shape={rejected_shape})",
                    flush=True,
                )

    sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
    manifest = {
        "dataset": "LocalLLaMA/typed-decisions (synthetic train augmentation)",
        "n_cases": int(sum(counts_by_wf.values())),
        "counts_by_workflow": dict(sorted(counts_by_wf.items())),
        "seed": seed,
        "state_author_model": author_model,
        "teacher_id": None,
        "shingle_k": shingle_k,
        "test_states_checked": n_test_states,
        "rejected_shingle_overlaps": rejected_shingle,
        "rejected_invalid_or_duplicate": rejected_shape,
        "cases_sha256": sha,
    }
    manifest_path = out_path.parent / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=6000, help="Total synthetic cases across the 4 workflows")
    parser.add_argument("--seed", type=int, default=7, help="Deterministic RNG seed")
    parser.add_argument("--out", default="data/td_synth/cases.jsonl", help="Output JSONL path")
    parser.add_argument("--author-model", default="claude-haiku-4-5", help="State-authoring LLM")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)

    manifest = generate_synthetic_cases(
        n_total=args.n,
        seed=args.seed,
        out_path=Path(args.out),
        author_model=args.author_model,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
