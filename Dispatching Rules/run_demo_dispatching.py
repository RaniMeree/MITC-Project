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

# ── point to the demo data folder ────────────────────────────────────────────
import dispatching_rules as dr

_HERE = os.path.dirname(__file__)
_DEMO_CANDIDATES = [
    os.path.join(_HERE, "Data", "Data", "Demo", "db_export"),
    os.path.join(_HERE, "..", "Data", "Data", "Demo", "db_export"),
]

DEMO_DIR = next((p for p in _DEMO_CANDIDATES if os.path.exists(p)), _DEMO_CANDIDATES[0])

if not os.path.exists(os.path.join(DEMO_DIR, "ManufacturingOrders.tsv")):
    raise FileNotFoundError(
        "Demo dataset not found. Expected ManufacturingOrders.tsv in one of: "
        + "; ".join(os.path.abspath(p) for p in _DEMO_CANDIDATES)
    )

# temporarily patch the DATA_DIR used inside dispatching_rules
dr.DATA_DIR = DEMO_DIR


# ─────────────────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────────────────

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

    # load the demo data
    jobs, meta, wc_df, wc_avail, wc_units = dr.load_and_preprocess()

    wc_names = dict(zip(wc_df["Id"], wc_df["Description"]))

    print()
    print("  Orders loaded:")
    order_names = {1001: "Order-A", 1002: "Order-B", 1003: "Order-C"}
    for oid in sorted(meta):
        label = order_names.get(oid, str(oid))
        m = meta[oid]
        print(f"    {label}  priority={int(m['priority'])}  "
              f"release={m['release_h']:.0f}h  "
              f"due={m['due_h']:.0f}h  "
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
            print(f"    {wc_names.get(m_id, m_id):<8}  →  {', '.join(users)}")

    # ── run all dispatching rules ─────────────────────────────────────────────
    results, best_rule = dr.compare_all_rules(jobs, meta, wc_units)

    base_date = pd.Timestamp("2024-01-01 00:00:00")

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
        print(f"  {'#':>3}  {'Machine':<8}  {'Order':<10}  {'Op':>4}  "
              f"{'Start':>19}  {'End':>19}  {'Prio':>5}  {'Status':>6}")
        print(f"  {'-'*3}  {'-'*8}  {'-'*10}  {'-'*4}  "
              f"{'-'*19}  {'-'*19}  {'-'*5}  {'-'*6}")

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
            print(f"  {idx:>3}  {mname:<8}  {olabel:<10}  {int(entry['op_num']):>4}  "
                  f"{str(start_dt)[:19]:>19}  {str(end_dt)[:19]:>19}  "
                  f"{prio:>5}  {late:>6}")

        # ── (B) schedule grouped by machine ──────────────────────────────────
        print()
        print(f"  Schedule by machine:")
        print(f"  {'Machine':<8}  {'Order':<10}  {'Op':>4}  "
              f"{'Start':>19}  {'End':>19}  {'Prio':>5}  {'Status':>6}")
        print(f"  {'-'*8}  {'-'*10}  {'-'*4}  "
              f"{'-'*19}  {'-'*19}  {'-'*5}  {'-'*6}")

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
            if prev_m != m_id:
                print()
                prev_m = m_id
            print(f"  {mname:<8}  {olabel:<10}  {int(entry['op_num']):>4}  "
                  f"{str(start_dt)[:19]:>19}  {str(end_dt)[:19]:>19}  "
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
                  f"finished={str(finish)[:19]}  {status}")

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
    ax.set_xlabel("Time (hours from start)", fontsize=11)
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
