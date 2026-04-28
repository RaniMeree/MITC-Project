"""
LLM-based Job Shop Scheduler
=============================

Uses a local Ollama LLM (e.g. llama3) to decide the order in which jobs
should be processed.  All machine, worker, and shift constraints come from
the existing dispatching_rules simulator — only the *job ranking* is done
by the LLM.

Setup
-----
1. Install Ollama:  https://ollama.com/download
2. Pull a model:    ollama pull llama3
3. Start Ollama:    ollama serve          (runs on http://localhost:11434)
4. Run this file:   python llm_scheduler.py

Dependencies (pip install):
    requests
    pandas
    numpy
"""

import sys
import os
import re
import json

import requests
import numpy as np

# ── make dispatching_rules importable from the sibling folder ──────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Dispatching Rules"))
import dispatching_rules as dr

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

OLLAMA_URL    = "http://localhost:11434/api/generate"
MODEL         = "llama3"   # change to whichever model you have pulled, e.g. "mistral"
REQUEST_TIMEOUT = 180      # seconds to wait for a response

# How many jobs to include in a single prompt.
# Larger = better decisions but slower / may exceed context window.
MAX_JOBS_PER_PROMPT = 30

SHOW_PROMPT   = True       # print the prompt and raw LLM reply


# ─────────────────────────────────────────────────────────────────────────────
# 1.  OLLAMA HELPER
# ─────────────────────────────────────────────────────────────────────────────

def ask_ollama(prompt: str, model: str = MODEL) -> str:
    """Send a prompt to Ollama and return the text response."""
    payload = {
        "model":   model,
        "prompt":  prompt,
        "stream":  False,
        "options": {"temperature": 0.0},   # deterministic output
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()["response"]


# ─────────────────────────────────────────────────────────────────────────────
# 2.  BUILD PROMPT
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt(jobs: dict, meta: dict) -> tuple:
    """
    Create a short, clear scheduling prompt.

    Returns
    -------
    prompt    : str  — text to send to the LLM
    alias_map : dict — short number (str) -> real order_id
    """
    # Sort by due date so the prompt reads naturally
    all_ids    = sorted(meta, key=lambda oid: meta[oid]["due_h"])
    chosen_ids = all_ids[:MAX_JOBS_PER_PROMPT]

    # Use short numbers (1, 2, 3 …) in the prompt to keep it compact
    alias_map = {str(i + 1): oid for i, oid in enumerate(chosen_ids)}

    rows = []
    for short_id, oid in alias_map.items():
        m     = meta[oid]
        n_ops = len(jobs.get(oid, []))
        rows.append(
            f"  Job {int(short_id):>3}:  priority={int(m['priority'])}  "
            f"due_in={m['due_h']:>7.1f} h  operations={n_ops}"
        )

    prompt = (
        "You are a manufacturing scheduling expert.\n"
        "Your objective is to minimise the number of late jobs.\n\n"
        "Field meanings:\n"
        "  priority  : 1 = most urgent,  9 = least urgent\n"
        "  due_in    : hours from now until the job must be finished\n"
        "  operations: number of sequential steps on different machines\n\n"
        "Jobs to schedule:\n"
        + "\n".join(rows)
        + "\n\n"
        "Task: rank these jobs from FIRST (process earliest) to LAST.\n"
        "Consider both urgency (due_in) and priority together.\n\n"
        "Output ONLY a JSON array of the job numbers in your recommended order.\n"
        "Example format: [3, 1, 5, 2, 4]\n"
        "No explanation, no extra text — just the JSON array."
    )

    return prompt, alias_map


# ─────────────────────────────────────────────────────────────────────────────
# 3.  PARSE LLM RESPONSE
# ─────────────────────────────────────────────────────────────────────────────

def parse_llm_order(response: str, alias_map: dict) -> list:
    """
    Extract the JSON array from the LLM response and map short IDs back
    to real order IDs.

    Returns a list of real order IDs in the LLM's suggested order.
    Any hallucinated / unrecognised numbers are silently skipped.
    """
    match = re.search(r"\[[\s\d,]+\]", response)
    if not match:
        raise ValueError(
            f"LLM did not return a JSON array.\nFull response:\n{response}"
        )
    short_list = json.loads(match.group())
    return [alias_map[str(s)] for s in short_list if str(s) in alias_map]


# ─────────────────────────────────────────────────────────────────────────────
# 4.  TRANSLATE LLM ORDER → PRIORITIES  →  SIMULATE
# ─────────────────────────────────────────────────────────────────────────────

def run_llm_scheduler(
    jobs, meta, wc_units=None, wc_workers=None, worker_info=None, day_absences=None
):
    """
    Full pipeline:
      1. Ask the LLM to rank the jobs.
      2. Convert the ranking into numeric priorities (1 = top priority).
      3. Run the event-driven simulator with the PRIO dispatching rule.
         → All machine capacity, worker competences, and shift constraints
           are enforced exactly as in the traditional rules.

    Returns (schedule, twt, completion) — same format as dr.simulate().
    """
    prompt, alias_map = build_prompt(jobs, meta)

    if SHOW_PROMPT:
        sep = "─" * 64
        print(f"\n{sep}\nPROMPT SENT TO LLM ({MODEL}):\n{sep}\n{prompt}\n{sep}")

    # ── Query the LLM ────────────────────────────────────────────────────────
    try:
        raw_response = ask_ollama(prompt)
    except requests.exceptions.ConnectionError:
        print(
            "\n[ERROR] Cannot connect to Ollama.\n"
            f"  Make sure Ollama is running:   ollama serve\n"
            f"  And the model is available:    ollama pull {MODEL}\n"
        )
        sys.exit(1)
    except requests.exceptions.Timeout:
        print(f"\n[ERROR] Ollama request timed out after {REQUEST_TIMEOUT} s.")
        sys.exit(1)

    if SHOW_PROMPT:
        print(f"LLM RAW RESPONSE:\n{raw_response}\n{'─'*64}")

    llm_order = parse_llm_order(raw_response, alias_map)
    print(f"  LLM ranked {len(llm_order)} jobs "
          f"(out of {len(meta)} total; rest get lowest priority).")

    # ── Assign LLM-derived numeric priorities ────────────────────────────────
    #   Rank 0 (first in list)  → priority 1.0 (highest)
    #   Rank n-1 (last in list) → priority 9.0 (lowest)
    llm_meta  = {oid: dict(m) for oid, m in meta.items()}
    n         = max(len(llm_order) - 1, 1)

    for rank, oid in enumerate(llm_order):
        llm_meta[oid]["priority"] = round((rank / n) * 8.0 + 1.0, 2)

    # Jobs the LLM did not mention → lowest priority
    mentioned = set(llm_order)
    for oid in llm_meta:
        if oid not in mentioned:
            llm_meta[oid]["priority"] = 9.0

    # ── Run the existing simulator with PRIO rule ────────────────────────────
    return dr.simulate(
        jobs, llm_meta, "PRIO",
        wc_units, wc_workers, worker_info, day_absences
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Load data ────────────────────────────────────────────────────────────
    print("Loading manufacturing data ...")
    jobs, meta, wc_df, wc_avail, wc_units, wc_workers, worker_info = \
        dr.load_and_preprocess(absent_workers=set())
    print(f"  {len(jobs)} orders  |  "
          f"{sum(len(v) for v in jobs.values())} operations  |  "
          f"{len({op['machine'] for ops in jobs.values() for op in ops})} work centres")

    # ── Run traditional dispatching rules as a baseline ──────────────────────
    print("\nRunning traditional dispatching rules (baseline) ...")
    dr.VERBOSE = False   # silence per-operation log during baseline runs

    trad_results = {}
    for rule in dr.RULES:
        sched, twt, comp = dr.simulate(
            jobs, meta, rule, wc_units, wc_workers, worker_info
        )
        late = sum(1 for oid, ct in comp.items() if ct > meta[oid]["due_h"])
        avg_tard = (
            float(np.mean([max(0.0, ct - meta[oid]["due_h"]) for oid, ct in comp.items()]))
            if comp else 0.0
        )
        trad_results[rule] = {"twt": twt, "late": late, "avg_tard": avg_tard}
        print(f"  {rule:<6}  TWT={twt:>10.1f} h  late={late}")

    # ── Run LLM scheduler ────────────────────────────────────────────────────
    print(f"\nQuerying local LLM ({MODEL}) for job ranking ...")
    llm_sched, llm_twt, llm_comp = run_llm_scheduler(
        jobs, meta, wc_units, wc_workers, worker_info
    )

    llm_late     = sum(1 for oid, ct in llm_comp.items() if ct > meta[oid]["due_h"])
    llm_avg_tard = (
        float(np.mean([max(0.0, ct - meta[oid]["due_h"]) for oid, ct in llm_comp.items()]))
        if llm_comp else 0.0
    )

    # ── Print comparison table ───────────────────────────────────────────────
    print()
    print("=" * 62)
    print("  Scheduling Results Comparison")
    print("=" * 62)
    print(f"  {'Method':<12}  {'TWT (h)':>12}  {'Late jobs':>10}  {'Avg tard (h)':>13}")
    print(f"  {'-'*12}  {'-'*12}  {'-'*10}  {'-'*13}")

    for rule, r in trad_results.items():
        print(f"  {rule:<12}  {r['twt']:>12.1f}  {r['late']:>10}  {r['avg_tard']:>13.1f}")

    print(f"  {'LLM (llama)':<12}  {llm_twt:>12.1f}  {llm_late:>10}  {llm_avg_tard:>13.1f}")
    print("=" * 62)

    best_trad_rule = min(trad_results, key=lambda r: trad_results[r]["twt"])
    best_trad_twt  = trad_results[best_trad_rule]["twt"]

    print()
    if llm_twt <= best_trad_twt:
        pct = ((best_trad_twt - llm_twt) / max(best_trad_twt, 1e-9)) * 100
        print(f"  LLM schedule is BETTER by {pct:.1f}%  "
              f"(vs best traditional rule: {best_trad_rule})")
    else:
        pct = ((llm_twt - best_trad_twt) / max(best_trad_twt, 1e-9)) * 100
        print(f"  LLM schedule is {pct:.1f}% worse than best traditional rule ({best_trad_rule}).")
        print("  Try a larger model or increase MAX_JOBS_PER_PROMPT for better results.")


if __name__ == "__main__":
    main()
