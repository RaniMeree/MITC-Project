"""
LLM-based Job Shop Scheduler
=============================

Asks a local Ollama LLM to decide the order in which jobs should be
processed, then runs a constraint-aware greedy scheduler to produce a
full, human-readable schedule table.

All machine capacity, worker competence, and shift constraints are
respected — the LLM only decides which jobs go first.

Setup
-----
1. Install Ollama : https://ollama.com/download
2. Pull a model   : ollama pull llama3
3. Run this file  : py llm_scheduler.py
   (Ollama will be started automatically if needed)

Dependencies: requests  pandas  numpy
"""

import sys
import os
import re
import json

import subprocess
import time
import shutil

import requests
import pandas as pd
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

# Common Windows install path for Ollama when it is not on PATH
_OLLAMA_FALLBACK = os.path.join(
    os.environ.get("LOCALAPPDATA", ""), "Programs", "Ollama", "ollama.exe"
)

# Jobs shown per prompt — keep ≤50 to stay within the context window
MAX_JOBS_PER_PROMPT = 50

SHOW_PROMPT = True   # print the prompt + raw LLM reply

DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "Dispatching Rules", "Data"
)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  OLLAMA HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _ollama_exe() -> str:
    """Return path to ollama executable (PATH or Windows fallback)."""
    found = shutil.which("ollama")
    if found:
        return found
    if os.path.isfile(_OLLAMA_FALLBACK):
        return _OLLAMA_FALLBACK
    return "ollama"   # let it fail with a clear message


def ensure_ollama_running():
    """Start Ollama server if it is not already reachable."""
    try:
        requests.get("http://localhost:11434", timeout=3)
        return   # already up
    except Exception:
        pass

    exe = _ollama_exe()
    print(f"  Starting Ollama ({exe}) ...")
    subprocess.Popen(
        [exe, "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    # wait up to 10 s for it to become ready
    for _ in range(20):
        time.sleep(0.5)
        try:
            requests.get("http://localhost:11434", timeout=2)
            print("  Ollama is ready.")
            return
        except Exception:
            pass
    print("  Warning: Ollama may not have started in time — trying anyway.")


def ask_ollama(prompt: str) -> str:
    """Send a prompt to Ollama and return the text response."""
    payload = {
        "model":   MODEL,
        "prompt":  prompt,
        "stream":  False,
        "options": {"temperature": 0.0},
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()["response"]


# ─────────────────────────────────────────────────────────────────────────────
# 2.  BUILD PROMPT
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt(jobs: dict, meta: dict) -> tuple:
    """
    Return (prompt_text, alias_map).
    alias_map: short number string -> real order_id
    """
    sorted_ids = sorted(meta, key=lambda oid: meta[oid]["due_h"])
    chosen     = sorted_ids[:MAX_JOBS_PER_PROMPT]
    alias_map  = {str(i + 1): oid for i, oid in enumerate(chosen)}

    rows = []
    for short_id, oid in alias_map.items():
        m = meta[oid]
        rows.append(
            f"  Job {int(short_id):>3}:  priority={int(m['priority'])}  "
            f"due_in={m['due_h']:>7.1f} h  "
            f"operations={len(jobs.get(oid, []))}"
        )

    prompt = (
        "You are a manufacturing scheduling expert.\n"
        "Your goal is to minimise the number of late jobs.\n\n"
        "Field meanings:\n"
        "  priority  : 1 = most urgent,  9 = least urgent\n"
        "  due_in    : hours until the deadline\n"
        "  operations: number of sequential steps the job needs\n\n"
        "Jobs:\n" + "\n".join(rows) + "\n\n"
        "Rank these jobs from FIRST (start earliest) to LAST.\n"
        "A job with a tighter deadline and higher urgency should go first.\n\n"
        "Output ONLY a JSON array of job numbers in your recommended order.\n"
        "Example: [3, 1, 5, 2, 4]\n"
        "No explanation — just the JSON array."
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
# 4.  APPLY LLM RANKING → SIMULATE → RETURN SCHEDULE
# ─────────────────────────────────────────────────────────────────────────────

def run_llm_schedule(jobs, meta, wc_units, wc_workers, worker_info):
    """
    1. Ask LLM to rank jobs.
    2. Convert ranking to numeric priorities (1.0 = highest, 9.0 = lowest).
    3. Run the constraint-aware simulator (PRIO rule) to assign real times.
    Returns (schedule, completion, llm_meta).
    """
    ensure_ollama_running()
    prompt, alias_map = build_prompt(jobs, meta)

    if SHOW_PROMPT:
        sep = "─" * 64
        print(f"\n{sep}\nPROMPT → LLM ({MODEL}):\n{sep}\n{prompt}\n{sep}")

    try:
        raw = ask_ollama(prompt)
    except requests.exceptions.ConnectionError:
        print(
            f"\n[ERROR] Cannot connect to Ollama.\n"
            f"  Install from: https://ollama.com/download\n"
            f"  Then run:     ollama pull {MODEL}\n"
        )
        sys.exit(1)
    except requests.exceptions.Timeout:
        print(f"\n[ERROR] Ollama timed out after {REQUEST_TIMEOUT} s.")
        sys.exit(1)

    if SHOW_PROMPT:
        print(f"LLM RESPONSE:\n{raw}\n{'─'*64}")

    llm_order = parse_llm_order(raw, alias_map)
    print(f"  LLM ranked {len(llm_order)} jobs.")

    # Convert rank → priority
    llm_meta = {oid: dict(m) for oid, m in meta.items()}
    n = max(len(llm_order) - 1, 1)
    for rank, oid in enumerate(llm_order):
        llm_meta[oid]["priority"] = round((rank / n) * 8.0 + 1.0, 2)
    for oid in llm_meta:
        if oid not in set(llm_order):
            llm_meta[oid]["priority"] = 9.0

    dr.VERBOSE = False
    schedule, _, completion = dr.simulate(
        jobs, llm_meta, "PRIO", wc_units, wc_workers, worker_info
    )
    return schedule, completion, llm_meta


# ─────────────────────────────────────────────────────────────────────────────
# 5.  PRINT SCHEDULE TABLE
# ─────────────────────────────────────────────────────────────────────────────

def fmt_dt(base_date, hours: float) -> str:
    dt = base_date + pd.Timedelta(hours=hours)
    return dt.strftime("%a %b %d %H:%M")


def print_schedule_table(schedule, meta, wc_df, base_date):
    """Print the schedule as a numbered table like the example."""
    wc_names = dict(zip(wc_df["Id"], wc_df["Description"]))
    sorted_sched = sorted(schedule, key=lambda x: (x["start"], str(x["machine"])))

    C_NUM = 4;  C_MCH = 14;  C_ORD = 12;  C_OP = 4
    C_WRK = 8;  C_DT  = 19;  C_PRO =  5;  C_STA = 6

    header = (
        f"{'#':>{C_NUM}}  {'Machine':<{C_MCH}}  {'Order':<{C_ORD}}  "
        f"{'Op':>{C_OP}}  {'Worker':<{C_WRK}}  "
        f"{'Start':<{C_DT}}  {'End':<{C_DT}}  "
        f"{'Prio':>{C_PRO}}  {'Status'}"
    )
    sep = (f"{'---':>{C_NUM}}  {'--------':<{C_MCH}}  {'------':<{C_ORD}}  "
           f"{'----':>{C_OP}}  {'--------':<{C_WRK}}  "
           f"{'-------------------':<{C_DT}}  {'-------------------':<{C_DT}}  "
           f"{'-----':>{C_PRO}}  {'------'}")

    print()
    print("=" * len(header))
    print("  LLM-Generated Schedule")
    print("=" * len(header))
    print(header)
    print(sep)

    for i, entry in enumerate(sorted_sched, start=1):
        oid    = entry["order_id"]
        m_name = wc_names.get(entry["machine"], str(entry["machine"]))
        worker = str(entry["worker"]) if entry["worker"] else "-"
        prio   = int(meta[oid]["priority"])
        status = "LATE" if entry["end"] > meta[oid]["due_h"] + 1e-6 else "OK"
        start  = fmt_dt(base_date, entry["start"])
        end    = fmt_dt(base_date, entry["end"])
        oid_str = str(oid)
        if len(oid_str) > C_ORD:
            oid_str = "\u2026" + oid_str[-(C_ORD - 1):]
        print(
            f"{i:>{C_NUM}}  {m_name:<{C_MCH}}  {oid_str:<{C_ORD}}  "
            f"{int(entry['op_num']):>{C_OP}}  {worker:<{C_WRK}}  "
            f"{start:<{C_DT}}  {end:<{C_DT}}  "
            f"{prio:>{C_PRO}}  {status}"
        )

    late = sum(1 for e in sorted_sched if e["end"] > meta[e["order_id"]]["due_h"] + 1e-6)
    print()
    print(f"  Total operations : {len(sorted_sched)}")
    print(f"  Late jobs        : {late}")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("Loading manufacturing data ...")
    jobs, meta, wc_df, wc_avail, wc_units, wc_workers, worker_info = \
        dr.load_and_preprocess(absent_workers=set())
    n_ops = sum(len(v) for v in jobs.values())
    n_wc  = len({op["machine"] for ops in jobs.values() for op in ops})
    print(f"  {len(jobs)} orders  |  {n_ops} operations  |  {n_wc} work centres")

    # Base date for readable datetime output
    orders_tmp = pd.read_csv(
        os.path.join(DATA_DIR, "ManufacturingOrders.tsv"), sep="\t"
    )
    orders_tmp["StartDate"] = pd.to_datetime(orders_tmp["StartDate"], errors="coerce")
    base_date = orders_tmp["StartDate"].min()

    print(f"\nQuerying local LLM ({MODEL}) for job ranking ...")
    schedule, completion, llm_meta = run_llm_schedule(
        jobs, meta, wc_units, wc_workers, worker_info
    )

    print_schedule_table(schedule, llm_meta, wc_df, base_date)


if __name__ == "__main__":
    main()
