"""
run_demo_dispatching.py
=======================
Runs the full dispatching_rules pipeline against the simplified
3-order demo dataset in Data/Data/Demo/db_export/

Demo dataset:
  Order 1001 | Priority 1 (HIGHEST) | due 22h after start | 6 operations
  Order 1002 | Priority 5 (MEDIUM)  | due 38h after start | 7 operations
  Order 1003 | Priority 9 (LOW)     | due 60h after start | 5 operations

Machine contention (several orders share the same machine):
  PRESS  (WC 1) → Order A op10,  Order B op10,  Order C op30
  GRIND  (WC 2) → Order A op20+50, Order B op30, Order C op10
  CHROME (WC 3) → Order A op30,  Order B op20+60, Order C op40
  RIVET  (WC 4) → Order A op40,  Order B op40,  Order C op20
  MILL   (WC 5) → Order A op60,  Order B op50+70, Order C op50
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import dispatching_rules as dr

_HERE = os.path.dirname(__file__)
_DEMO_CANDIDATES = [
    os.path.join(_HERE, "Data"),
    os.path.join(_HERE, "..", "Data"),
    os.path.join(_HERE, "Data", "Data", "Demo", "db_export"),
]

DEMO_DIR = next((p for p in _DEMO_CANDIDATES if os.path.exists(p)), _DEMO_CANDIDATES[0])

if not os.path.exists(os.path.join(DEMO_DIR, "ManufacturingOrders.tsv")):
    raise FileNotFoundError(
        "Demo dataset not found. Expected ManufacturingOrders.tsv in one of: "
        + "; ".join(os.path.abspath(p) for p in _DEMO_CANDIDATES)
    )

# temporarily patch the DATA_DIR used inside dispatching_rules
dr.DATA_DIR = DEMO_DIR


# RUN the solution
if __name__ == "__main__":

    import sys

    OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "demo_schedule_output.txt")

    # write to both terminal AND output file at the same time
    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, text):
            for f in self.files:
                f.write(text)
        def flush(self):
            for f in self.files:
                f.flush()

    out_f = open(OUTPUT_FILE, "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, out_f)

    print("=" * 65)
    print("  DEMO DISPATCHING — 3 orders, 5 machines, 18 operations")
    print("=" * 65)

    # ── ask for absent workers then load everything ───────────────────────────
    print()
    raw = input("Enter absent worker IDs (comma-separated), or press Enter to skip: ").strip()
    absent_workers = set()
    if raw:
        for token in raw.split(","):
            t = token.strip()
            if t:
                absent_workers.add(t)

    if absent_workers:
        print(f"  Absent workers : {sorted(absent_workers)}")
    else:
        print("  No absent workers.")

    # load data — competences are built from WorkerCompetences.csv with absent excluded
    jobs, meta, wc_df, wc_avail, wc_units, wc_workers, worker_info = dr.load_and_preprocess(absent_workers)

    wc_names = dict(zip(wc_df["Id"], wc_df["Description"]))
    TODAY = pd.Timestamp.now().normalize()  # today at midnight — Day 1 of the schedule

    # show which machines are available / blocked
    print()
    print("  Worker availability per machine (competence 2 required):")
    for m_id, mname in sorted(wc_names.items()):
        workers = wc_workers.get(m_id, [])
        if m_id in wc_workers:
            if workers:
                shifts = ", ".join(
                    f"W{w} {worker_info[w]['shift_start']:.0f}:00-{worker_info[w]['shift_end']:.0f}:00"
                    for w in workers if w in worker_info
                )
                print(f"    WC {m_id} {mname:<8}  workers: {', '.join(workers)}  ({shifts})")
            else:
                print(f"    WC {m_id} {mname:<8}  *** NO WORKER AVAILABLE — operations will be SKIPPED ***")
        else:
            print(f"    WC {m_id} {mname:<8}  (no worker constraint)")

    print()
    print("  Orders loaded:")
    order_names = {1001: "Order-A", 1002: "Order-B", 1003: "Order-C"}
    for oid in sorted(meta):
        label = order_names.get(oid, str(oid))
        m = meta[oid]
        release_dt = TODAY + pd.Timedelta(hours=m['release_h'])
        due_dt     = TODAY + pd.Timedelta(hours=m['due_h'])
        print(f"    {label}  priority={int(m['priority'])}  "
              f"release={release_dt.strftime('%b %d %H:%M')}  "
              f"due={due_dt.strftime('%b %d %H:%M')}  "
              f"ops={len(jobs[oid])}")

    print()
    print("  Machine contention (shared across orders):")
    from collections import defaultdict
    machine_users = defaultdict(list)
    for oid, ops in jobs.items():
        for op in ops:
            machine_users[op["machine"]].append(
                f"{order_names.get(oid, str(oid))}-op{int(op['op_num'])}"
            )
    for m_id, users in sorted(machine_users.items()):
        if len(users) > 1:
            print(f"    {wc_names.get(m_id, m_id):<8}  ->  {', '.join(users)}")

    # ── run all dispatching rules ─────────────────────────────────────────────
    results, best_rule = dr.compare_all_rules(jobs, meta, wc_units, wc_workers, worker_info)

    base_date = TODAY

    # ── full schedule + operation order for EVERY rule ────────────────────────
    for rule in dr.RULES:
        print()
        print("=" * 75)
        print(f"  RULE: {rule}  —  TWT={results[rule]['twt']:.1f}h  "
              f"late={results[rule]['late']}  "
              f"{'<-- BEST' if rule == best_rule else ''}")
        print("=" * 75)

        sched = sorted(results[rule]["schedule"], key=lambda x: x["start"])
        comp  = results[rule]["completion"]

        # ── (A) chronological operation order ────────────────────────────────
        print()
        print(f"  Operation execution order (chronological):")
        print(f"  {'#':>3}  {'Machine':<8}  {'Order':<10}  {'Op':>4}  {'Worker':<8}  "
              f"{'Start':>14}  {'End':>14}  {'Prio':>5}  {'Status':>6}")
        print(f"  {'-'*3}  {'-'*8}  {'-'*10}  {'-'*4}  {'-'*8}  "
              f"{'-'*14}  {'-'*14}  {'-'*5}  {'-'*6}")

        for idx, entry in enumerate(sched, 1):
            m_id     = entry["machine"]
            oid      = entry["order_id"]
            prio     = int(meta[oid]["priority"])
            due_h    = meta[oid]["due_h"]
            late     = "LATE" if entry["end"] > due_h else "OK"
            start_dt = base_date + pd.Timedelta(hours=entry["start"])
            end_dt   = base_date + pd.Timedelta(hours=entry["end"])
            mname    = wc_names.get(m_id, str(m_id))
            olabel   = order_names.get(oid, str(oid))
            worker   = str(entry.get("worker") or "-")
            print(f"  {idx:>3}  {mname:<8}  {olabel:<10}  {int(entry['op_num']):>4}  {worker:<8}  "
                  f"{start_dt.strftime('%b %d %H:%M'):>14}  {end_dt.strftime('%b %d %H:%M'):>14}  "
                  f"{prio:>5}  {late:>6}")

        # ── (B) schedule grouped by machine ──────────────────────────────────
        print()
        print(f"  Schedule by machine:")
        print(f"  {'Machine':<8}  {'Order':<10}  {'Op':>4}  {'Worker':<8}  "
              f"{'Start':>14}  {'End':>14}  {'Prio':>5}  {'Status':>6}")
        print(f"  {'-'*8}  {'-'*10}  {'-'*4}  {'-'*8}  "
              f"{'-'*14}  {'-'*14}  {'-'*5}  {'-'*6}")

        by_machine = {}
        for entry in sched:
            by_machine.setdefault(entry["machine"], []).append(entry)

        prev_m = None
        for entry in sorted(sched, key=lambda x: (x["machine"], x["start"])):
            m_id     = entry["machine"]
            oid      = entry["order_id"]
            prio     = int(meta[oid]["priority"])
            due_h    = meta[oid]["due_h"]
            late     = "LATE" if entry["end"] > due_h else "OK"
            start_dt = base_date + pd.Timedelta(hours=entry["start"])
            end_dt   = base_date + pd.Timedelta(hours=entry["end"])
            mname    = wc_names.get(m_id, str(m_id))
            olabel   = order_names.get(oid, str(oid))
            worker   = str(entry.get("worker") or "-")
            if prev_m != m_id:
                print()
                prev_m = m_id
            print(f"  {mname:<8}  {olabel:<10}  {int(entry['op_num']):>4}  {worker:<8}  "
                  f"{start_dt.strftime('%b %d %H:%M'):>14}  {end_dt.strftime('%b %d %H:%M'):>14}  "
                  f"{prio:>5}  {late:>6}")

        # ── (C) order summary ─────────────────────────────────────────────────
        print()
        print(f"  Order summary:")
        for oid in sorted(comp):
            ct     = comp[oid]
            due    = meta[oid]["due_h"]
            prio   = int(meta[oid]["priority"])
            tard   = max(0.0, ct - due)
            status = f"LATE by {tard:.1f}h" if tard > 0 else "ON TIME"
            finish = base_date + pd.Timedelta(hours=ct)
            print(f"    {order_names.get(oid, str(oid)):<10}  "
                  f"priority={prio}  "
                  f"finished={finish.strftime('%b %d %H:%M')}  {status}")

    # ── final comparison ──────────────────────────────────────────────────────
    print()
    print("=" * 75)
    print("  FINAL COMPARISON")
    print("=" * 75)
    print(f"  {'Rule':<6}  {'TWT (h)':>10}  {'Late':>6}  {'Makespan':>10}  {'Winner':>8}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*6}  {'-'*10}  {'-'*8}")
    for rule in dr.RULES:
        twt  = results[rule]["twt"]
        late = results[rule]["late"]
        comp = results[rule]["completion"]
        mksp = max(comp.values()) if comp else 0
        star = "<-- BEST" if rule == best_rule else ""
        print(f"  {rule:<6}  {twt:>10.1f}  {late:>6}  {mksp:>10.1f}  {star}")
    print()

    # restore stdout and close file
    sys.stdout = sys.__stdout__
    out_f.close()
    print(f"\n  Output saved to: {OUTPUT_FILE}")

    # ── Gantt chart for the best rule ────────────────────────────────────────
    ORDER_COLOURS = {
        1001: "#4C72B0",   # blue   – Order-A
        1002: "#DD8452",   # orange – Order-B
        1003: "#55A868",   # green  – Order-C
    }

    schedule   = results[best_rule]["schedule"]
    completion = results[best_rule]["completion"]

    machine_order, seen = [], set()
    for entry in sorted(schedule, key=lambda x: x["start"]):
        m = entry["machine"]
        if m not in seen:
            machine_order.append(m)
            seen.add(m)

    machine_y  = {m: i for i, m in enumerate(machine_order)}
    n_machines = len(machine_order)

    fig, ax = plt.subplots(figsize=(14, max(4, n_machines * 1.2)))
    BAR_HEIGHT = 0.55

    # shift window for day shading (use first worker's shift, all same in demo)
    _sh_start = 8.0
    _sh_end   = 16.0
    if worker_info:
        _fi = next(iter(worker_info.values()))
        _sh_start = _fi["shift_start"]
        _sh_end   = _fi["shift_end"]

    _max_end   = max((e["end"] for e in schedule), default=24.0) + 8.0
    _day_count = int(_max_end / 24) + 2

    # shade off-shift periods (before shift start and after shift end each day)
    for _d in range(_day_count):
        ax.axvspan(_d*24,           _d*24 + _sh_start, alpha=0.10, color="gray", zorder=0)
        ax.axvspan(_d*24 + _sh_end, (_d+1)*24,         alpha=0.10, color="gray", zorder=0)

    for entry in schedule:
        m     = entry["machine"]
        oid   = entry["order_id"]
        y     = machine_y[m]
        start = entry["start"]
        dur   = entry["end"] - entry["start"]
        color = ORDER_COLOURS.get(oid, "#999999")
        late  = entry["end"] > meta[oid]["due_h"]

        ax.barh(y, dur, left=start, height=BAR_HEIGHT,
                color=color, edgecolor="white", linewidth=0.8, alpha=0.9)

        label = f"{order_names.get(oid, str(oid)).split('-')[1]}-op{int(entry['op_num'])}"
        ax.text(start + dur / 2, y, label,
                ha="center", va="center", fontsize=7.5,
                color="white", fontweight="bold")

        if late:
            ax.barh(y, dur, left=start, height=BAR_HEIGHT,
                    color="none", edgecolor="red", linewidth=2.0)

    for oid, m_info in meta.items():
        due   = m_info["due_h"]
        color = ORDER_COLOURS.get(oid, "#999999")
        ax.axvline(due, color=color, linestyle="--", linewidth=1.2, alpha=0.6)
        ax.text(due, n_machines - 0.3,
                f"{order_names.get(oid, str(oid))}\ndue",
                ha="center", va="bottom", fontsize=7, color=color)

    ax.set_yticks(range(n_machines))
    ax.set_yticklabels([wc_names.get(m, str(m)) for m in machine_order], fontsize=10)

    # X-axis: one tick per working day, labelled "Day N / Date"
    _work_ticks  = [_d*24 + _sh_start for _d in range(_day_count)
                    if _d*24 + _sh_start <= _max_end]
    _work_labels = [
        f"Day {_d+1}\n{(TODAY + pd.Timedelta(days=_d)).strftime('%b %d')}"
        for _d in range(len(_work_ticks))
    ]
    ax.set_xticks(_work_ticks)
    ax.set_xticklabels(_work_labels, fontsize=9)
    ax.set_xlim(0, _max_end)
    ax.set_xlabel("Working Day", fontsize=11)
    ax.set_title(
        f"Gantt Chart — Best Rule: {best_rule}  "
        f"(TWT = {results[best_rule]['twt']:.1f} h, "
        f"late orders = {results[best_rule]['late']})",
        fontsize=12, fontweight="bold",
    )
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.6)

    patches = [
        mpatches.Patch(color=ORDER_COLOURS[oid],
                       label=f"{order_names[oid]}  (prio={int(meta[oid]['priority'])})")
        for oid in sorted(ORDER_COLOURS)
    ]
    patches.append(mpatches.Patch(facecolor="none", edgecolor="red",
                                   linewidth=2, label="LATE finish"))
    ax.legend(handles=patches, loc="lower right", fontsize=9)

    plt.tight_layout()
    chart_path = os.path.join(_HERE, "gantt_chart.png")
    plt.savefig(chart_path, dpi=150)
    print(f"  Gantt chart saved to: {chart_path}")
    plt.show()
