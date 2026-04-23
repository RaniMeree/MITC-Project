

import os
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(
    os.path.dirname(__file__),
    "Data", "Data", "Ztift 2025", "db_export"
)

DEFAULT_DURATION = 8.0   # fallback duration in hours if data is missing

RULES = ["EDD", "SPT", "WSPT", "PRIO", "FIFO", "CR"]
# WSPT = Weighted Shortest Processing Time: combines priority + duration


# ─────────────────────────────────────────────────────────────────────────────
# 1a.  LOAD WORK CENTRES
# ─────────────────────────────────────────────────────────────────────────────

def load_work_centres():
    """
    Read WorkCenters.tsv and return:
      wc_df    : raw DataFrame (kept for display / printing)
      wc_avail : dict  machine_id -> availability factor (0..1)
                 e.g. 80% available = 0.80 → durations will be divided by 0.80
      wc_units : dict  machine_id -> number of parallel planning slots
                 e.g. 2 means two identical machines can run at the same time
    """
    wc_df = pd.read_csv(os.path.join(DATA_DIR, "WorkCenters.tsv"), sep="\t")

    # availability factor: 80 in the data → 0.80
    wc_avail = dict(zip(
        wc_df["Id"],
        wc_df["AvailabilityFactor"].fillna(100.0) / 100.0
    ))

    # number of parallel slots per machine (default 1)
    wc_units = {
        row["Id"]: max(1, int(round(float(row.get("NumberOfPlanningUnits", 1) or 1))))
        for _, row in wc_df.iterrows()
    }

    return wc_df, wc_avail, wc_units


# ─────────────────────────────────────────────────────────────────────────────
# 1b.  LOAD MANUFACTURING ORDERS
# ─────────────────────────────────────────────────────────────────────────────

def load_orders():
    """
    Read ManufacturingOrders.tsv and return a processed DataFrame with:
      release_h : hours since the earliest StartDate (= order becomes available)
      due_h     : hours since the earliest StartDate (= order must be finished)
      Priority  : kept as-is from the source (1 = high, 9 = low)

    Orders with missing dates are dropped.
    """
    orders_df = pd.read_csv(os.path.join(DATA_DIR, "ManufacturingOrders.tsv"), sep="\t")

    orders_df["StartDate"]  = pd.to_datetime(orders_df["StartDate"],  errors="coerce")
    orders_df["FinishDate"] = pd.to_datetime(orders_df["FinishDate"], errors="coerce")

    # use the earliest StartDate as time-zero so all hours are relative
    base = orders_df["StartDate"].min()
    orders_df["release_h"] = (orders_df["StartDate"]  - base).dt.total_seconds() / 3600
    orders_df["due_h"]     = (orders_df["FinishDate"] - base).dt.total_seconds() / 3600

    # drop orders where dates are missing
    orders_df = orders_df.dropna(subset=["release_h", "due_h"])

    return orders_df


# ─────────────────────────────────────────────────────────────────────────────
# 1c.  LOAD MANUFACTURING OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

def load_operations(wc_avail):
    """
    Read ManufacturingOperations.tsv and return a processed DataFrame with:
      duration_h : planned duration in hours
                   = (PlannedFinishDate - PlannedStartDate)
                   adjusted by machine availability factor
                   clipped to minimum 0.5 h

    wc_avail is passed in so we can adjust duration for machine availability.
    A machine that is only 80% available takes 1.25× longer to finish work.
    """
    ops_df = pd.read_csv(
        os.path.join(DATA_DIR, "ManufacturingOperations.tsv"),
        sep="\t", low_memory=False
    )

    ops_df["PlannedStartDate"]  = pd.to_datetime(ops_df["PlannedStartDate"],  errors="coerce")
    ops_df["PlannedFinishDate"] = pd.to_datetime(ops_df["PlannedFinishDate"], errors="coerce")

    # duration from planned dates
    planned_hours = (
        (ops_df["PlannedFinishDate"] - ops_df["PlannedStartDate"])
        .dt.total_seconds() / 3600
    )
    ops_df["duration_h"] = planned_hours.fillna(DEFAULT_DURATION)
    ops_df["duration_h"] = ops_df["duration_h"].clip(lower=0.5)  # at least 30 min

    # adjust for machine availability: less available = effectively longer
    ops_df["avail"]      = ops_df["WorkCenterId"].map(wc_avail).fillna(1.0)
    ops_df["duration_h"] = ops_df["duration_h"] / ops_df["avail"].clip(lower=0.1)

    return ops_df


# ─────────────────────────────────────────────────────────────────────────────
# 1d.  BUILD jobs AND meta DICTS  (joins the three tables)
# ─────────────────────────────────────────────────────────────────────────────

def build_jobs_and_meta(orders_df, ops_df):
    """
    Join operations with their parent order and build:

      jobs : dict  order_id -> list of operation dicts sorted by OperationNumber
             Each op dict: { op_num, machine, duration }

      meta : dict  order_id -> { release_h, due_h, priority }
             Contains the order-level info used by dispatching rules.

    Only orders that have BOTH operations and order metadata are kept.
    """
    # join each operation row with its parent order's dates and priority
    ops = (
        ops_df
        .merge(
            orders_df[["Id", "release_h", "due_h", "Priority"]],
            left_on="ManufacturingOrderId",
            right_on="Id",
            suffixes=("", "_order")
        )
        .sort_values(["ManufacturingOrderId", "OperationNumber"])
    )

    # build jobs dict: one entry per order, value = ordered list of operations
    jobs = {}
    for order_id, grp in ops.groupby("ManufacturingOrderId"):
        jobs[order_id] = [
            {
                "op_num":   row["OperationNumber"],
                "machine":  row["WorkCenterId"],
                "duration": row["duration_h"],
            }
            for _, row in grp.iterrows()
        ]

    # build meta dict: order-level info (release, due, priority)
    meta = {}
    for _, row in orders_df[orders_df["Id"].isin(jobs)].iterrows():
        prio = row["Priority"]
        meta[row["Id"]] = {
            "release_h": float(row["release_h"]),
            "due_h":     float(row["due_h"]),
            "priority":  float(prio) if pd.notna(prio) else 5.0,
        }

    # keep only orders that appear in both dicts
    common = set(jobs) & set(meta)
    jobs   = {k: jobs[k] for k in common}
    meta   = {k: meta[k] for k in common}

    return jobs, meta


# ─────────────────────────────────────────────────────────────────────────────
# 1.  MAIN ENTRY POINT FOR LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_and_preprocess():
    """
    Orchestrates the four loading steps and returns everything the
    simulator needs.
    """
    wc_df, wc_avail, wc_units = load_work_centres()
    orders_df                 = load_orders()
    ops_df                    = load_operations(wc_avail)
    jobs, meta                = build_jobs_and_meta(orders_df, ops_df)

    return jobs, meta, wc_df, wc_avail, wc_units


# DISPATCHING RULES
def pick_next(queue, current_time, rule):
    """
    Given a list of ready operations (queue), pick the one
    that the dispatching rule says goes next.
    """
    if rule == "EDD":
        # Earlies due date
        return min(queue, key=lambda x: x["due"])

    elif rule == "SPT":
        # Operation with the shortest processing time
        return min(queue, key=lambda x: x["op"]["duration"])

    elif rule == "PRIO":
        # Lower priority number = higher importance
        return min(queue, key=lambda x: x["prio"])

    elif rule == "FIFO":
        # Job that was released / started earliest goes first
        return min(queue, key=lambda x: x["rel"])

    elif rule == "WSPT":
        # Weighted Shortest Processing Time: duration / weight
        # Combines BOTH priority AND processing time — higher priority short jobs go first
        def wspt(x):
            weight = max(1.0, 10.0 - x["prio"])  # priority 1 → weight 9, priority 9 → weight 1
            return x["op"]["duration"] / weight
        return min(queue, key=wspt)

    elif rule == "CR":
        # Critical Ratio = time remaining / work remaining
        # Smallest CR = most behind schedule → goes first
        def cr(x):
            time_left = x["due"] - current_time
            work_left = x["op"]["duration"]
            return time_left / (work_left + 1e-9)
        return min(queue, key=cr)

    raise ValueError(f"Unknown rule: {rule}")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  EVENT-DRIVEN SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────

def simulate(jobs, meta, rule, wc_units=None):
    """
    Simulate the job shop using an event-driven approach.

    Each machine runs independently.  At each event (a machine becomes free),
    the dispatching rule decides which waiting operation goes next on that machine.

    Returns
    -------
    schedule        : list of dicts  {order_id, op_num, machine, start, end}
    twt             : Total Weighted Tardiness in hours
    completion_time : dict  order_id -> finish time of last operation
    """

    # ── initialise state ─────────────────────────────────────────────────────
    next_op_idx = {oid: 0 for oid in jobs}
    job_done_at = {oid: meta[oid]["release_h"] for oid in jobs}

    mach_free  = {}   # machine -> list of free-times, one entry per planning unit
    mach_queue = {}   # machine -> list of waiting items

    for oid, ops in jobs.items():
        for op in ops:
            m = op["machine"]
            if m not in mach_free:
                units = 1 if wc_units is None else max(1, wc_units.get(m, 1))
                mach_free[m]  = [0.0] * units   # one slot per planning unit
                mach_queue[m] = []

    def enqueue(oid):
        """Add the next unscheduled operation of a job to its machine queue."""
        idx = next_op_idx[oid]
        if idx < len(jobs[oid]):
            op       = jobs[oid][idx]
            ready_at = max(job_done_at[oid], meta[oid]["release_h"])
            mach_queue[op["machine"]].append({
                "oid":      oid,
                "op":       op,
                "ready_at": ready_at,
                "due":      meta[oid]["due_h"],
                "prio":     meta[oid]["priority"],
                "rel":      meta[oid]["release_h"],
            })

    # put the first operation of every job into the appropriate machine queue
    for oid in jobs:
        enqueue(oid)

    schedule       = []
    completion     = {}
    total_ops      = sum(len(v) for v in jobs.values())
    done           = 0

    # ── main simulation loop ──────────────────────────────────────────────────
    while done < total_ops:

        # Find the machine whose next possible start time is earliest
        best_time = float("inf")
        best_m    = None
        best_slot = None

        for m, q in mach_queue.items():
            if not q:
                continue
            # pick the slot (planning unit) on this machine that is free soonest
            earliest_ready = min(e["ready_at"] for e in q)
            slot_idx = min(range(len(mach_free[m])), key=lambda i: mach_free[m][i])
            t = max(mach_free[m][slot_idx], earliest_ready)
            if t < best_time:
                best_time = t
                best_m    = m
                best_slot = slot_idx

        if best_m is None:
            break  # no more work to schedule

        t = best_time
        q = mach_queue[best_m]

        # All operations in this machine's queue that are ready by time t
        ready_now = [e for e in q if e["ready_at"] <= t + 1e-9]

        # Apply the dispatching rule to pick one
        chosen = pick_next(ready_now, t, rule)
        q.remove(chosen)

        oid   = chosen["oid"]
        op    = chosen["op"]
        start = max(chosen["ready_at"], mach_free[best_m][best_slot])
        end   = start + op["duration"]

        # commit the decision
        mach_free[best_m][best_slot] = end
        job_done_at[oid]  = end
        next_op_idx[oid] += 1

        schedule.append({
            "order_id": oid,
            "op_num":   op["op_num"],
            "machine":  best_m,
            "start":    start,
            "end":      end,
        })
        done += 1

        if next_op_idx[oid] == len(jobs[oid]):
            completion[oid] = end   # last op of this job is done
        else:
            enqueue(oid)            # put next op in its machine queue

    # ── compute Total Weighted Tardiness ─────────────────────────────────────
    twt = 0.0
    for oid, ct in completion.items():
        due      = meta[oid]["due_h"]
        prio     = meta[oid]["priority"]
        # lower priority number = more important job  →  higher weight
        weight   = max(1.0, 10.0 - prio)
        tardiness = max(0.0, ct - due)
        twt      += weight * tardiness

    return schedule, twt, completion


# ─────────────────────────────────────────────────────────────────────────────
# 4.  COMPARE ALL RULES
# ─────────────────────────────────────────────────────────────────────────────

def compare_all_rules(jobs, meta, wc_units=None):
    """Run every dispatching rule and print a comparison table."""

    print()
    print("=" * 60)
    print("  Dispatching Rule Comparison — Ztift 2025 Data")
    print("=" * 60)
    machines = {op["machine"] for ops in jobs.values() for op in ops}
    print(f"  Orders       : {len(jobs)}")
    print(f"  Operations   : {sum(len(v) for v in jobs.values())}")
    print(f"  Work Centres : {len(machines)}")
    print("=" * 60)
    print(f"  {'Rule':<6}  {'TWT (hours)':>14}  {'Late orders':>12}  {'Avg tardiness (h)':>18}")
    print(f"  {'-'*6}  {'-'*14}  {'-'*12}  {'-'*18}")

    results = {}

    for rule in RULES:
        schedule, twt, completion = simulate(jobs, meta, rule, wc_units)

        late_orders = [
            oid for oid, ct in completion.items()
            if ct > meta[oid]["due_h"]
        ]
        late_count = len(late_orders)

        avg_tard = np.mean([
            max(0.0, ct - meta[oid]["due_h"])
            for oid, ct in completion.items()
        ]) if completion else 0.0

        results[rule] = {
            "twt":      twt,
            "late":     late_count,
            "avg_tard": avg_tard,
            "schedule": schedule,
            "completion": completion,
        }

        print(f"  {rule:<6}  {twt:>14.1f}  {late_count:>12}  {avg_tard:>18.1f}")

    best = min(results, key=lambda r: results[r]["twt"])
    print()
    print(f"  Best rule : {best}  "
          f"(TWT = {results[best]['twt']:.1f} h, "
          f"late orders = {results[best]['late']})")
    print("=" * 60)

    return results, best


# ─────────────────────────────────────────────────────────────────────────────
# 5.  PRINT SCHEDULE
# ─────────────────────────────────────────────────────────────────────────────

def print_schedule(schedule, meta, wc_df, base_date, rule, top_machines=5):
    """
    Print a human-readable schedule grouped by work centre.
    Shows real datetime (converted back from float hours).
    Only prints the top_machines busiest work centres to keep output manageable.
    """
    wc_names = dict(zip(wc_df["Id"], wc_df["Description"]))

    # group by machine
    from collections import defaultdict
    by_machine = defaultdict(list)
    for entry in schedule:
        by_machine[entry["machine"]].append(entry)

    # sort each machine's ops by start time
    for m in by_machine:
        by_machine[m].sort(key=lambda x: x["start"])

    # pick the busiest machines (most operations)
    busiest = sorted(by_machine, key=lambda m: len(by_machine[m]), reverse=True)[:top_machines]

    print()
    print("=" * 70)
    print(f"  Schedule output — rule: {rule}  (top {top_machines} busiest work centres)")
    print("=" * 70)

    for m in busiest:
        name = wc_names.get(m, str(m))
        ops  = by_machine[m]
        print(f"\n  Work Centre: {name}  ({len(ops)} operations)")
        print(f"  {'Order':>8}  {'Op':>4}  {'Start':>22}  {'End':>22}  {'Priority':>8}  {'Status':>8}")
        print(f"  {'-'*8}  {'-'*4}  {'-'*22}  {'-'*22}  {'-'*8}  {'-'*8}")
        for entry in ops:
            oid   = entry["order_id"]
            prio  = int(meta[oid]["priority"])
            due_h = meta[oid]["due_h"]
            late  = "LATE" if entry["end"] > due_h else "OK"
            # convert float hours → real datetime
            start_dt = base_date + pd.Timedelta(hours=entry["start"])
            end_dt   = base_date + pd.Timedelta(hours=entry["end"])
            print(f"  {oid % 100000:>8}  {int(entry['op_num']):>4}  "
                  f"{str(start_dt)[:19]:>22}  {str(end_dt)[:19]:>22}  "
                  f"{prio:>8}  {late:>8}")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# 6.  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading Ztift 2025 data ...")
    jobs, meta, wc_df, wc_avail = load_and_preprocess()
    print(f"  {len(jobs)} orders with "
          f"{sum(len(v) for v in jobs.values())} operations loaded")

    results, best_rule = compare_all_rules(jobs, meta)

    # ── print the actual schedule for the best rule ───────────────────────────

    orders_df_tmp = pd.read_csv(
        os.path.join(DATA_DIR, "ManufacturingOrders.tsv"), sep="\t"
    )
    orders_df_tmp["StartDate"] = pd.to_datetime(orders_df_tmp["StartDate"], errors="coerce")
    base_date = orders_df_tmp["StartDate"].min()

    print_schedule(
        results[best_rule]["schedule"],
        meta,
        wc_df,
        base_date,
        rule=best_rule,
        top_machines=5
    )
