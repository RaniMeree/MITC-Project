import os, re, json
import requests
import pandas as pd

DATA       = os.path.join(os.path.dirname(__file__), "..", "Dispatching Rules", "Data")
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL      = "llama3"
SHIFT_S, SHIFT_E = 8.0, 17.0   # 08:00 - 17:00 (1-hour grace)


def next_shift(t):
    day = int(t // 24)
    tod = t - day * 24
    if tod < SHIFT_S:  return day * 24 + SHIFT_S
    if tod >= SHIFT_E: return (day + 1) * 24 + SHIFT_S
    return t


def load_data():
    orders = pd.read_csv(f"{DATA}/ManufacturingOrders.tsv", sep="\t",
                         parse_dates=["StartDate", "FinishDate"])
    ops    = pd.read_csv(f"{DATA}/ManufacturingOperations.tsv", sep="\t", low_memory=False)
    wc     = pd.read_csv(f"{DATA}/WorkCenters.tsv", sep="\t")
    comp   = pd.read_csv(f"{DATA}/WorkerCompetences.csv")

    base = pd.Timestamp.now().normalize()
    orders["release_h"] = (orders["StartDate"]  - base).dt.total_seconds() / 3600
    orders["due_h"]     = (orders["FinishDate"] - base).dt.total_seconds() / 3600
    orders = orders.dropna(subset=["release_h", "due_h"])

    ops["duration_h"] = (
        (ops["PlannedQuantity"].fillna(0) * ops["PlannedUnitTime"].fillna(0)
         + ops["PlannedSetupTime"].fillna(0)) / 1e9 / 3600
    ).clip(lower=0.5)

    merged = ops.merge(orders[["Id", "release_h", "due_h", "Priority"]],
                       left_on="ManufacturingOrderId", right_on="Id")

    jobs, meta = {}, {}
    for oid, grp in merged.groupby("ManufacturingOrderId"):
        grp = grp.sort_values("OperationNumber")
        jobs[oid] = [{"op_num": r["OperationNumber"], "machine": r["WorkCenterId"],
                      "duration": r["duration_h"]} for _, r in grp.iterrows()]
        meta[oid] = {"release_h": float(grp["release_h"].iloc[0]),
                     "due_h":     float(grp["due_h"].iloc[0]),
                     "priority":  float(grp["Priority"].iloc[0])}

    # Build machine -> [comp-2 worker IDs]
    wc_num_to_id = dict(zip(wc["Number"].astype(int), wc["Id"]))
    worker_col   = comp.columns[0]
    comp_long    = comp.melt(id_vars=worker_col, var_name="wc_num", value_name="level")
    comp_long["level"] = pd.to_numeric(comp_long["level"], errors="coerce").fillna(0)
    wc_workers = {}
    for _, row in comp_long[comp_long["level"] == 2].iterrows():
        wc_id = wc_num_to_id.get(int(row["wc_num"]))
        if wc_id:
            wc_workers.setdefault(wc_id, []).append(str(row[worker_col]))

    return jobs, meta, dict(zip(wc["Id"], wc["Description"])), wc_workers, base


def build_prompt(jobs, meta):
    alias_map = {str(i + 1): oid for i, oid in enumerate(meta)}
    rows = [
        f"  Job {int(k):>3}: priority={int(meta[oid]['priority'])}  "
        f"due_in={meta[oid]['due_h']:>7.1f}h  operations={len(jobs.get(oid, []))}"
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


def schedule_jobs(llm_order, jobs, meta, wc_workers):
    machine_free, worker_free = {}, {}
    schedule = []
    for oid in llm_order:
        t = max(meta[oid]["release_h"], 0.0)
        for op in jobs[oid]:
            m     = op["machine"]
            start = next_shift(max(t, machine_free.get(m, 0.0)))
            workers = wc_workers.get(m, [])
            if workers:
                best_w = min(workers, key=lambda w: next_shift(worker_free.get(w, 0.0)))
                start  = next_shift(max(start, worker_free.get(best_w, 0.0)))
            else:
                best_w = None
            # defer if op would finish more than 1h past shift end
            day = int(start // 24)
            if start + op["duration"] > day * 24 + SHIFT_E + 1.0:
                start = next_shift(day * 24 + SHIFT_E + 0.001)
            end = start + op["duration"]
            machine_free[m] = end
            if best_w:
                worker_free[best_w] = end
            t = end
            schedule.append({"order_id": oid, "op_num": op["op_num"],
                              "machine": m, "worker": best_w, "start": start, "end": end})
    return schedule


if __name__ == "__main__":
    jobs, meta, wc_names, wc_workers, base = load_data()

    prompt, alias_map = build_prompt(jobs, meta)
    print(f"PROMPT:\n{prompt}\n")

    raw = requests.post(OLLAMA_URL, json={"model": MODEL, "prompt": prompt,
                        "stream": False, "options": {"temperature": 0.0}}).json()["response"]
    print(f"LLM RESPONSE:\n{raw}\n")

    match     = re.search(r"\[[\s\d,]+\]", raw)
    llm_order = [alias_map[str(s)] for s in json.loads(match.group()) if str(s) in alias_map]
    llm_order += [oid for oid in meta if oid not in set(llm_order)]

    schedule = schedule_jobs(llm_order, jobs, meta, wc_workers)

    def fmt(h):
        return (base + pd.Timedelta(hours=h)).strftime("%a %b %d %H:%M")

    print(f"  {'#':>4}  {'Machine':<14}  {'Order':<12}  {'Op':>4}  {'Worker':<8}  {'Start':<18}  {'End':<18}  {'Prio':>5}  Status")
    print(f"  {'-'*4}  {'-'*14}  {'-'*12}  {'-'*4}  {'-'*8}  {'-'*18}  {'-'*18}  {'-'*5}  ------")
    for i, e in enumerate(sorted(schedule, key=lambda x: x["start"]), 1):
        oid    = e["order_id"]
        status = "LATE" if e["end"] > meta[oid]["due_h"] else "OK"
        print(f"  {i:>4}  {wc_names.get(e['machine'], str(e['machine'])):<14}  "
              f"{str(oid):<12}  {int(e['op_num']):>4}  {str(e['worker'] or '-'):<8}  "
              f"{fmt(e['start']):<18}  {fmt(e['end']):<18}  "
              f"{int(meta[oid]['priority']):>5}  {status}")

    late = sum(1 for e in schedule if e["end"] > meta[e["order_id"]]["due_h"])
    print(f"\n  Total operations: {len(schedule)}   Late jobs: {late}")