import sys, os, re, json, subprocess, time, shutil
import requests
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Dispatching Rules"))
import dispatching_rules as dr

OLLAMA_URL  = "http://localhost:11434/api/generate"
MODEL       = "llama3"
TIMEOUT     = 180
MAX_JOBS    = 50
_OLLAMA_EXE = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Ollama", "ollama.exe")


def ensure_ollama_running():
    try:
        requests.get("http://localhost:11434", timeout=3)
        return
    except Exception:
        pass
    exe = shutil.which("ollama") or _OLLAMA_EXE
    print(f"Starting Ollama ({exe}) ...")
    subprocess.Popen([exe, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    for _ in range(20):
        time.sleep(0.5)
        try:
            requests.get("http://localhost:11434", timeout=2)
            return
        except Exception:
            pass


def ask_ollama(prompt):
    r = requests.post(OLLAMA_URL, json={"model": MODEL, "prompt": prompt,
                      "stream": False, "options": {"temperature": 0.0}}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()["response"]


def build_prompt(jobs, meta):
    chosen    = sorted(meta, key=lambda oid: meta[oid]["due_h"])[:MAX_JOBS]
    alias_map = {str(i + 1): oid for i, oid in enumerate(chosen)}
    rows      = [
        f"  Job {int(k):>3}:  priority={int(meta[oid]['priority'])}  "
        f"due_in={meta[oid]['due_h']:>7.1f} h  operations={len(jobs.get(oid, []))}"
        for k, oid in alias_map.items()
    ]
    prompt = (
        "You are a manufacturing scheduling expert and your job is to minimize the latency of the orders.\n\n"
        "Rules:\n"
        "  - Job priorities span from 1 to 9 where 1 is the least and 9 is the most urgent.\n"
        "  - Consider each order's deadline (due_in: hours until the deadline).\n"
        "  - Operations must run sequentially: an operation cannot start until the previous operation in the same order is done.\n"
        "  - A work centre can process only one job at a time.\n"
        "  - A work centre needs a competent worker (competence level 2) to perform an operation.\n"
        "  - Workers and work centres operate only between 08:00 and 16:00.\n"
        "  - Orders with a tighter deadline and higher urgency should go first.\n"
        "  - If two jobs have the same priority, prefer the one with the closer deadline (smaller due_in).\n\n"
        "Field meanings:\n"
        "  priority  : 9 = most urgent,  1 = least urgent\n"
        "  due_in    : hours until the deadline\n"
        "  operations: number of sequential steps the job needs\n\n"
        "Jobs:\n" + "\n".join(rows) + "\n\n"
        "Rank these jobs from FIRST (start earliest) to LAST.\n\n"
        "Output ONLY a JSON array of job numbers in your recommended order.\n"
        "Example: [3, 1, 5, 2, 4]\n"
        "No explanation - just the JSON array."
    )
    return prompt, alias_map


def run_llm_schedule(jobs, meta, wc_units, wc_workers, worker_info, start_weekday=0):
    ensure_ollama_running()
    prompt, alias_map = build_prompt(jobs, meta)

    print(f"\nPROMPT:\n{'-'*60}\n{prompt}\n{'-'*60}")
    raw = ask_ollama(prompt)
    print(f"LLM RESPONSE:\n{raw}\n{'-'*60}")

    match = re.search(r"\[[\s\d,]+\]", raw)
    if not match:
        raise ValueError(f"LLM did not return a JSON array.\n{raw}")
    llm_order = [alias_map[str(s)] for s in json.loads(match.group()) if str(s) in alias_map]
    print(f"LLM ranked {len(llm_order)} jobs.")

    # Convert rank position to priority value (rank 0 -> priority 1.0, last -> 9.0)
    llm_meta = {oid: dict(m) for oid, m in meta.items()}
    n = max(len(llm_order) - 1, 1)
    for rank, oid in enumerate(llm_order):
        llm_meta[oid]["priority"] = round((rank / n) * 8.0 + 1.0, 2)
    for oid in llm_meta:
        if oid not in set(llm_order):
            llm_meta[oid]["priority"] = 9.0

    dr.VERBOSE = False
    schedule, _, completion = dr.simulate(
        jobs, llm_meta, "PRIO", wc_units, wc_workers, worker_info,
        start_weekday=start_weekday
    )
    return schedule, completion, llm_meta


def print_schedule(schedule, meta, wc_df, base_date):
    wc_names     = dict(zip(wc_df["Id"], wc_df["Description"]))
    sorted_sched = sorted(schedule, key=lambda x: x["start"])

    def fmt(h):
        return (base_date + pd.Timedelta(hours=h)).strftime("%a %b %d %H:%M")

    print()
    print(f"  {'#':>4}  {'Machine':<14}  {'Order':<12}  {'Op':>4}  {'Worker':<8}  {'Start':<18}  {'End':<18}  {'Prio':>5}  Status")
    print(f"  {'-'*4}  {'-'*14}  {'-'*12}  {'-'*4}  {'-'*8}  {'-'*18}  {'-'*18}  {'-'*5}  ------")

    for i, e in enumerate(sorted_sched, 1):
        oid    = e["order_id"]
        status = "LATE" if e["end"] > meta[oid]["due_h"] + 1e-6 else "OK"
        print(f"  {i:>4}  {wc_names.get(e['machine'], str(e['machine'])):<14}  "
              f"{str(oid):<12}  {int(e['op_num']):>4}  {str(e['worker'] or '-'):<8}  "
              f"{fmt(e['start']):<18}  {fmt(e['end']):<18}  "
              f"{int(meta[oid]['priority']):>5}  {status}")

    late = sum(1 for e in sorted_sched if e["end"] > meta[e["order_id"]]["due_h"] + 1e-6)
    print(f"\n  Total operations: {len(sorted_sched)}   Late jobs: {late}")


if __name__ == "__main__":
    print("Loading data ...")
    jobs, meta, wc_df, _, wc_units, wc_workers, worker_info = dr.load_and_preprocess(absent_workers=set())
    print(f"  {len(jobs)} orders, {sum(len(v) for v in jobs.values())} operations")

    base_date = pd.Timestamp.now().normalize()
    schedule, _, llm_meta = run_llm_schedule(jobs, meta, wc_units, wc_workers, worker_info,
                                             start_weekday=base_date.weekday())
    print_schedule(schedule, llm_meta, wc_df, base_date)