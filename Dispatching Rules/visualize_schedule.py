"""
visualize_schedule.py
=====================
Generates a Gantt chart for the best dispatching rule on the demo dataset.

Each row = one machine (work centre)
Each bar = one operation, coloured by order
X axis   = time in hours from start
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── locate demo data the same way run_demo_dispatching does ──────────────────
import dispatching_rules as dr

_HERE = os.path.dirname(__file__)
_DEMO_CANDIDATES = [
    os.path.join(_HERE, "Data", "Data", "Demo", "db_export"),
    os.path.join(_HERE, "..", "Data", "Data", "Demo", "db_export"),
]
DEMO_DIR = next((p for p in _DEMO_CANDIDATES if os.path.exists(p)), _DEMO_CANDIDATES[0])
dr.DATA_DIR = DEMO_DIR

# ── load demo data and run all rules ─────────────────────────────────────────
jobs, meta, wc_df, wc_avail, wc_units = dr.load_and_preprocess()
results, best_rule = dr.compare_all_rules(jobs, meta, wc_units)

schedule   = results[best_rule]["schedule"]
completion = results[best_rule]["completion"]

# ── display names ─────────────────────────────────────────────────────────────
order_names  = {1001: "Order-A", 1002: "Order-B", 1003: "Order-C"}
wc_names     = dict(zip(wc_df["Id"], wc_df["Description"]))

# colours per order
ORDER_COLOURS = {
    1001: "#4C72B0",   # blue  – Order-A (highest priority)
    1002: "#DD8452",   # orange – Order-B
    1003: "#55A868",   # green – Order-C
}

# ── collect machine list in order of first appearance ─────────────────────────
machine_order = []
seen = set()
for entry in sorted(schedule, key=lambda x: x["start"]):
    m = entry["machine"]
    if m not in seen:
        machine_order.append(m)
        seen.add(m)

machine_y = {m: i for i, m in enumerate(machine_order)}
n_machines = len(machine_order)

# ── build chart ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, max(4, n_machines * 1.2)))

BAR_HEIGHT = 0.55

for entry in schedule:
    m     = entry["machine"]
    oid   = entry["order_id"]
    y     = machine_y[m]
    start = entry["start"]
    dur   = entry["end"] - entry["start"]
    color = ORDER_COLOURS.get(oid, "#999999")
    due   = meta[oid]["due_h"]
    late  = entry["end"] > due

    # operation bar
    ax.barh(
        y,
        dur,
        left=start,
        height=BAR_HEIGHT,
        color=color,
        edgecolor="white",
        linewidth=0.8,
        alpha=0.9,
    )

    # label inside bar: "A-op10"
    label = f"{order_names.get(oid, str(oid)).split('-')[1]}-op{int(entry['op_num'])}"
    ax.text(
        start + dur / 2,
        y,
        label,
        ha="center",
        va="center",
        fontsize=7.5,
        color="white",
        fontweight="bold",
    )

    # red border for LATE finish
    if late:
        ax.barh(
            y,
            dur,
            left=start,
            height=BAR_HEIGHT,
            color="none",
            edgecolor="red",
            linewidth=2.0,
        )

# ── due-date vertical lines ───────────────────────────────────────────────────
for oid, m_info in meta.items():
    due   = m_info["due_h"]
    color = ORDER_COLOURS.get(oid, "#999999")
    ax.axvline(
        due,
        color=color,
        linestyle="--",
        linewidth=1.2,
        alpha=0.6,
        label=f"{order_names.get(oid, str(oid))} due",
    )
    ax.text(due, n_machines - 0.3, f"{order_names.get(oid, str(oid))}\ndue",
            ha="center", va="bottom", fontsize=7, color=color)

# ── axes labels ───────────────────────────────────────────────────────────────
ax.set_yticks(range(n_machines))
ax.set_yticklabels([wc_names.get(m, str(m)) for m in machine_order], fontsize=10)
ax.set_xlabel("Time (hours from start)", fontsize=11)
ax.set_title(
    f"Gantt Chart — Best Rule: {best_rule}  "
    f"(TWT = {results[best_rule]['twt']:.1f} h, "
    f"late orders = {results[best_rule]['late']})",
    fontsize=12,
    fontweight="bold",
)
ax.invert_yaxis()   # top machine first
ax.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.6)

# ── legend ────────────────────────────────────────────────────────────────────
patches = [
    mpatches.Patch(color=ORDER_COLOURS[oid], label=f"{order_names[oid]}  (prio={int(meta[oid]['priority'])})")
    for oid in sorted(ORDER_COLOURS)
]
patches.append(mpatches.Patch(facecolor="none", edgecolor="red", linewidth=2, label="LATE finish"))
ax.legend(handles=patches, loc="lower right", fontsize=9)

plt.tight_layout()

out_path = os.path.join(_HERE, "gantt_chart.png")
plt.savefig(out_path, dpi=150)
print(f"\n  Gantt chart saved to: {out_path}")
plt.show()
