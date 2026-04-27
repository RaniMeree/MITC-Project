

import os
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(
    os.path.dirname(__file__),
    "..", "Data"
)

DEFAULT_DURATION = 8.0   # fallback duration in hours if data is missing

RULES = ["EDD", "SPT", "WSPT", "PRIO", "FIFO", "CR"]
# WSPT = Weighted Shortest Processing Time: combines priority + duration

VERBOSE  = True   # set to False to silence all debug prints
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "simulation.log")


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
# 1c.  LOAD WORKER COMPETENCES
# ─────────────────────────────────────────────────────────────────────────────

def _parse_hhmm(s):
    """Parse 'HH:MM' string to float hours (e.g. '8:00' -> 8.0, '16:30' -> 16.5)."""
    h, m = str(s).strip().split(":")
    return int(h) + int(m) / 60.0


def load_competences(absent_workers=None):
    """
    Read WorkerCompetences.csv and return:
      wc_workers  : dict  wc_id -> [worker_ids with competence 2, not absent]
                    Every machine that has ANY competence-2 worker is included
                    (empty list = all absent → BLOCKED).
      worker_info : dict  worker_id -> {shift_start: float, shift_end: float}
                    Shift hours are 0-24 wall-clock hours (e.g. 8.0, 16.0).

    WorkerCompetences.csv uses WC *Number* (100, 120 …); we translate to
    WC Id (1-5) using WorkCenters.tsv.
    """
    if absent_workers is None:
        absent_workers = set()

    wc_df = pd.read_csv(os.path.join(DATA_DIR, "WorkCenters.tsv"), sep="\t")
    number_to_id = dict(zip(wc_df["Number"].astype(int), wc_df["Id"].astype(int)))

    comp_df = pd.read_csv(os.path.join(DATA_DIR, "WorkerCompetences.csv"))
    comp_df["CompetenceLevel"] = comp_df["CompetenceLevel"].fillna(0).astype(int)
    comp_df["WorkerId"]        = comp_df["WorkerId"].fillna("").astype(str)

    # parse shift times (default 00:00 – 24:00 if columns missing)
    if "starting" in comp_df.columns and "finish" in comp_df.columns:
        comp_df["shift_start"] = comp_df["starting"].apply(_parse_hhmm)
        comp_df["shift_end"]   = comp_df["finish"].apply(_parse_hhmm)
    else:
        comp_df["shift_start"] = 0.0
        comp_df["shift_end"]   = 24.0

    # build worker_info for ALL workers (competence 1 and 2), not just absent ones
    worker_info = {}
    for _, row in comp_df.drop_duplicates("WorkerId").iterrows():
        worker_info[str(row["WorkerId"])] = {
            "shift_start": float(row["shift_start"]),
            "shift_end":   float(row["shift_end"]),
        }

    # competence-2 only rows
    comp2 = comp_df[comp_df["CompetenceLevel"] == 2]

    # initialise every machine that has competence-2 workers (empty = BLOCKED)
    wc_workers = {}
    for _, row in comp2.iterrows():
        wc_id = number_to_id.get(int(row["WorkCenterId"]), int(row["WorkCenterId"]))
        if wc_id not in wc_workers:
            wc_workers[wc_id] = []

    # fill in non-absent workers
    for _, row in comp2.iterrows():
        w = str(row["WorkerId"])
        if w not in absent_workers:
            wc_id = number_to_id.get(int(row["WorkCenterId"]), int(row["WorkCenterId"]))
            wc_workers[wc_id].append(w)

    return wc_workers, worker_info


def next_shift_start(t, shift_start_h, shift_end_h):
    """
    Given a simulation time t (float hours since base date, day 0 = today),
    return the earliest time >= t at which the worker is on shift.

    shift_start_h / shift_end_h are wall-clock hours (e.g. 8.0, 16.0).
    The shift repeats every 24 hours.  If shift_end <= shift_start the shift
    is treated as 24 h (no restriction).
    """
    if shift_end_h <= shift_start_h:          # no real restriction
        return t
    shift_len = shift_end_h - shift_start_h   # e.g. 8 h

    day   = int(t // 24)                      # which calendar day
    tod   = t - day * 24                      # time-of-day (0-24)

    on_today_start = day * 24 + shift_start_h
    on_today_end   = day * 24 + shift_end_h

    if tod < shift_start_h:
        return on_today_start          # worker starts later today
    if tod < shift_end_h:
        return t                       # worker is currently on shift
    # past today's shift — wait for tomorrow
    return on_today_start + 24.0


def next_available_time(t, shift_start_h, shift_end_h, absent_weekdays=None):
    """
    Like next_shift_start but also skips:
      - weekends  (day_num % 7 >= 5)
      - days when the worker is absent  (weekday in absent_weekdays)

    Day 0 of the simulation is assumed to be Monday.
    absent_weekdays : set of 0-4  (0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri)
                      meaning the worker does not come in on those days.
    Absence only applies to the first work-week (day_num 0-6).
    """
    if absent_weekdays is None:
        absent_weekdays = set()
    if shift_end_h <= shift_start_h:   # treat as no shift limit
        shift_start_h, shift_end_h = 0.0, 24.0

    for _ in range(60):                # safety cap — max 60 days of searching
        day_num = int(t // 24)
        weekday = day_num % 7          # 0=Mon … 6=Sun
        tod     = t - day_num * 24     # current time-of-day

        # skip weekends — jump to next Monday's shift start
        if weekday >= 5:
            days_ahead = 7 - weekday   # Sat→2, Sun→1
            t = (day_num + days_ahead) * 24 + shift_start_h
            continue

        # skip absent days (only enforce for the first 7-day window)
        if day_num < 7 and weekday in absent_weekdays:
            t = (day_num + 1) * 24
            continue

        # snap to shift-start if too early
        if tod < shift_start_h:
            return day_num * 24 + shift_start_h

        # past today's shift — try tomorrow
        if tod >= shift_end_h:
            t = (day_num + 1) * 24
            continue

        return t   # worker is available right now

    return t   # fallback


# 1d.  LOAD MANUFACTURING OPERATIONS

def load_operations(wc_avail):
    """
    Read ManufacturingOperations.tsv and return a processed DataFrame with:
      duration_h : operation duration in hours
                   = (RestQuantity * ReportedUnitTime + ReportedSetupTime) / 1e9 / 3600
                   (time fields are stored in nanoseconds)
                   adjusted by machine availability factor
                   clipped to minimum 0.5 h

    wc_avail is passed in so we can adjust duration for machine availability.
    A machine that is only 80% available takes 1.25× longer to finish work.
    """
    ops_df = pd.read_csv(
        os.path.join(DATA_DIR, "ManufacturingOperations.tsv"),
        sep="\t", low_memory=False
    )

    # duration = (RestQuantity * ReportedUnitTime + ReportedSetupTime) in nanoseconds → hours
    rest_qty   = ops_df["RestQuantity"].fillna(0)
    unit_time  = ops_df["ReportedUnitTime"].fillna(0)
    setup_time = ops_df["ReportedSetupTime"].fillna(0)
    duration_ns = rest_qty * unit_time + setup_time
    ops_df["duration_h"] = duration_ns / 1e9 / 3600

    # fall back to DEFAULT_DURATION if result is zero or NaN
    ops_df["duration_h"] = ops_df["duration_h"].replace(0, DEFAULT_DURATION).fillna(DEFAULT_DURATION)
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

def load_and_preprocess(absent_workers=None):
    """
    Orchestrates the loading steps and returns everything the simulator needs.
    absent_workers : set of worker ID strings to exclude.
                     Pass None to prompt interactively (default).
                     Pass an empty set() to skip silently.
    """
    wc_df, wc_avail, wc_units = load_work_centres()
    orders_df                 = load_orders()
    ops_df                    = load_operations(wc_avail)
    jobs, meta                = build_jobs_and_meta(orders_df, ops_df)

    if absent_workers is None:
        # interactive fallback when called directly
        print()
        raw = input("Enter absent worker IDs (comma-separated), or press Enter to skip: ").strip()
        absent_workers = set()
        if raw:
            for token in raw.split(","):
                token = token.strip()
                if token:
                    absent_workers.add(token)
        if absent_workers:
            print(f"  Absent workers: {sorted(absent_workers)}")
        else:
            print("  No absent workers.")

    wc_workers, worker_info = load_competences(absent_workers)

    return jobs, meta, wc_df, wc_avail, wc_units, wc_workers, worker_info


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

def simulate(jobs, meta, rule, wc_units=None, wc_workers=None, worker_info=None, day_absences=None):
    """
    Simulate the job shop using an event-driven approach.

    wc_workers   : dict  machine_id -> [worker_ids with competence 2]
    worker_info  : dict  worker_id  -> {shift_start, shift_end}  (wall-clock hours)
                   Operations can only start when a qualified worker is on shift.
    day_absences : dict  worker_id  -> set of weekday indices (0=Mon … 4=Fri)
                   Worker is not available on those days of the first work-week.

    Returns
    -------
    schedule        : list of dicts  {order_id, op_num, machine, worker, start, end}
    twt             : Total Weighted Tardiness in hours
    completion_time : dict  order_id -> finish time of last operation
    """
    if wc_workers is None:
        wc_workers = {}
    if worker_info is None:
        worker_info = {}

    # open log file (append so all 6 rules land in one file per run)
    _log_fh = open(LOG_PATH, "a", encoding="utf-8") if VERBOSE else None
    def _log(*args, **kwargs):
        if _log_fh:
            print(*args, file=_log_fh, flush=True, **kwargs)

    # ── initialise state ─────────────────────────────────────────────────────
    next_op_idx = {oid: 0 for oid in jobs}
    job_done_at = {oid: meta[oid]["release_h"] for oid in jobs}

    mach_free  = {}   # machine -> list of free-times, one entry per planning unit
    mach_queue = {}   # machine -> list of waiting items
    worker_free = {}  # worker_id -> time when the worker is next free

    for oid, ops in jobs.items():
        for op in ops:
            m = op["machine"]
            if m not in mach_free:
                units = 1 if wc_units is None else max(1, wc_units.get(m, 1))
                mach_free[m]  = [0.0] * units   # one slot per planning unit
                mach_queue[m] = []

    if VERBOSE:
        _log(f"\n{'='*65}")
        _log(f"[SIM] Rule: {rule}")
        _log(f"[SIM] Orders: {len(jobs)}  Total ops: {sum(len(v) for v in jobs.values())}")
        _log(f"[SIM] Machines: {sorted(mach_free.keys())}")
        for m, slots in mach_free.items():
            workers = wc_workers.get(m, [])
            units   = len(slots)
            _log(f"       machine={m}  slots={units}  workers={workers}")
        _log(f"[SIM] Job releases / due dates / priority:")
        for oid in jobs:
            m_info = meta[oid]
            _log(f"       order={oid}  release={m_info['release_h']:.1f}h  due={m_info['due_h']:.1f}h  prio={m_info['priority']}  ops={len(jobs[oid])}")
        _log(f"{'='*65}")

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
            if VERBOSE:
                _log(f"[SIM]   ENQUEUE order={oid} op#{op['op_num']} -> machine={op['machine']}  ready@{ready_at:.2f}h  duration={op['duration']:.2f}h")

    # put the first operation of every job into the appropriate machine queue
    if VERBOSE:
        _log(f"\n[SIM] Enqueuing first operation of each order:")
    for oid in jobs:
        enqueue(oid)

    schedule       = []
    completion     = {}
    total_ops      = sum(len(v) for v in jobs.values())
    done           = 0
    step           = 0

    if VERBOSE:
        _log(f"\n[SIM] Starting simulation loop ({total_ops} operations total)...")

    # ── main simulation loop ──────────────────────────────────────────────────
    while done < total_ops:
        step += 1
        if VERBOSE:
            print(f"\n[SIM] ── Step {step}  (done={done}/{total_ops}) ──")
            for m, q in mach_queue.items():
                if q:
                    q_str = ", ".join(f"order={e['oid']} op#{e['op']['op_num']} ready@{e['ready_at']:.1f}h" for e in q)
                    _log(f"       machine={m} queue: [{q_str}]")
                else:
                    _log(f"       machine={m} queue: [empty]")

        # Find the machine whose next possible start time is earliest
        best_time = float("inf")
        best_m    = None
        best_slot = None

        for m, q in mach_queue.items():
            if not q:
                continue
            # skip machines where we have worker data but all eligible workers are absent
            eligible = wc_workers.get(m, [])
            if m in wc_workers and not eligible:
                if VERBOSE:
                    print(f"       machine={m} SKIPPED (no available workers)")
                continue  # machine is tracked but no worker with competence 2 available
            # earliest time a worker for this machine is free (shift + absence aware)
            if eligible:
                worker_avail = min(
                    next_available_time(
                        worker_free.get(w, 0.0),
                        worker_info.get(w, {}).get("shift_start", 0.0) if worker_info else 0.0,
                        worker_info.get(w, {}).get("shift_end",  24.0) if worker_info else 24.0,
                        day_absences.get(w, set()) if day_absences else set()
                    )
                    for w in eligible
                )
            else:
                worker_avail = 0.0
            # pick the slot (planning unit) on this machine that is free soonest
            earliest_ready = min(e["ready_at"] for e in q)
            slot_idx = min(range(len(mach_free[m])), key=lambda i: mach_free[m][i])
            t = max(mach_free[m][slot_idx], earliest_ready, worker_avail)
            if VERBOSE:
                _log(f"       machine={m}: mach_free_slot={mach_free[m][slot_idx]:.2f}h  earliest_ready={earliest_ready:.2f}h  worker_avail={worker_avail:.2f}h  -> candidate_t={t:.2f}h")
            if t < best_time:
                best_time = t
                best_m    = m
                best_slot = slot_idx

        if best_m is None:
            break  # no more work to schedule

        if VERBOSE:
            _log(f"[SIM]   >> Selected machine={best_m}  slot={best_slot}  earliest_start={best_time:.2f}h")

        t = best_time
        q = mach_queue[best_m]

        # All operations in this machine's queue that are ready by time t
        ready_now = [e for e in q if e["ready_at"] <= t + 1e-9]

        if VERBOSE:
            print(f"[SIM]   Ready queue for machine={best_m} at t={t:.2f}h:")
            for e in ready_now:
                _log(f"         order={e['oid']} op#{e['op']['op_num']} due={e['due']:.1f}h prio={e['prio']} duration={e['op']['duration']:.2f}h ready@{e['ready_at']:.2f}h")

        # Apply the dispatching rule to pick one
        chosen = pick_next(ready_now, t, rule)
        q.remove(chosen)

        if VERBOSE:
            _log(f"[SIM]   Rule={rule} picked: order={chosen['oid']} op#{chosen['op']['op_num']}")

        oid   = chosen["oid"]
        op    = chosen["op"]

        # pick the eligible worker who can start soonest (shift + absence aware)
        eligible = wc_workers.get(best_m, [])
        if VERBOSE:
            _log(f"[SIM]   Eligible workers for machine={best_m}: {eligible}")
        if eligible:
            def worker_earliest(w):
                """Earliest time worker w is free, on shift, and not absent."""
                free_at  = worker_free.get(w, 0.0)
                info     = worker_info.get(w, {}) if worker_info else {}
                ss       = info.get("shift_start", 0.0)
                se       = info.get("shift_end",   24.0)
                abs_days = day_absences.get(w, set()) if day_absences else set()
                return next_available_time(free_at, ss, se, abs_days)
            if VERBOSE:
                for w in eligible:
                    _log(f"         worker={w}  free@{worker_free.get(w,0.0):.2f}h  shift={worker_info.get(w,{}).get('shift_start',0):.1f}-{worker_info.get(w,{}).get('shift_end',24):.1f}  -> avail@{worker_earliest(w):.2f}h")
            best_worker       = min(eligible, key=worker_earliest)
            worker_start_free = worker_earliest(best_worker)
            if VERBOSE:
                _log(f"[SIM]   >> Chosen worker: {best_worker}  available@{worker_start_free:.2f}h")
        else:
            best_worker = None
            worker_start_free = 0.0
            if VERBOSE:
                _log(f"[SIM]   No workers assigned to machine={best_m}, no constraint")

        # also snap the machine/job ready-time to the worker's shift + absence window
        tentative = max(chosen["ready_at"], mach_free[best_m][best_slot])
        if best_worker:
            info     = worker_info.get(best_worker, {}) if worker_info else {}
            ss       = info.get("shift_start", 0.0)
            se       = info.get("shift_end",   24.0)
            abs_days = day_absences.get(best_worker, set()) if day_absences else set()
            tentative = next_available_time(tentative, ss, se, abs_days)
        start = max(tentative, worker_start_free)
        end   = start + op["duration"]

        if VERBOSE:
            _log(f"[SIM]   SCHEDULE: order={oid} op#{op['op_num']} machine={best_m} worker={best_worker}")
            _log(f"                   tentative={tentative:.2f}h  worker_free={worker_start_free:.2f}h")
            _log(f"                   START={start:.2f}h  END={end:.2f}h  duration={op['duration']:.2f}h")

        # commit the decision
        mach_free[best_m][best_slot] = end
        if best_worker is not None:
            worker_free[best_worker] = end
        job_done_at[oid]  = end
        next_op_idx[oid] += 1

        if VERBOSE:
            _log(f"[SIM]   State update: machine={best_m} slot={best_slot} now free@{end:.2f}h")
            if best_worker:
                _log(f"                     worker={best_worker} now free@{end:.2f}h")
            _log(f"                     job_done_at[{oid}]={end:.2f}h  next_op_idx={next_op_idx[oid]}/{len(jobs[oid])}")

        schedule.append({
            "order_id": oid,
            "op_num":   op["op_num"],
            "machine":  best_m,
            "worker":   best_worker,
            "start":    start,
            "end":      end,
        })
        done += 1

        if next_op_idx[oid] == len(jobs[oid]):
            completion[oid] = end   # last op of this job is done
            if VERBOSE:
                _log(f"[SIM]   *** ORDER {oid} COMPLETE at {end:.2f}h (due={meta[oid]['due_h']:.2f}h) ***")
        else:
            enqueue(oid)            # put next op in its machine queue

    # ── compute Total Weighted Tardiness ─────────────────────────────────────
    if VERBOSE:
        _log(f"\n[SIM] ── TWT Calculation ──")
    twt = 0.0
    for oid, ct in completion.items():
        due       = meta[oid]["due_h"]
        prio      = meta[oid]["priority"]
        # lower priority number = more important job  →  higher weight
        weight    = max(1.0, 10.0 - prio)
        tardiness = max(0.0, ct - due)
        contrib   = weight * tardiness
        twt      += contrib
        if VERBOSE:
            status = f"LATE by {tardiness:.2f}h" if tardiness > 0 else "on time"
            _log(f"       order={oid}  finish={ct:.2f}h  due={due:.2f}h  prio={prio}  weight={weight:.1f}  tardiness={tardiness:.2f}h  contrib={contrib:.2f}  [{status}]")
    if VERBOSE:
        _log(f"[SIM] Total TWT = {twt:.4f}h")
        _log(f"{'='*65}\n")
        _log_fh.close()

    return schedule, twt, completion


# ─────────────────────────────────────────────────────────────────────────────
# 4.  COMPARE ALL RULES
# ─────────────────────────────────────────────────────────────────────────────

def compare_all_rules(jobs, meta, wc_units=None, wc_workers=None, worker_info=None, day_absences=None):
    """Run every dispatching rule and print a comparison table."""

    # clear the log file at the start of each full run
    if VERBOSE:
        open(LOG_PATH, "w", encoding="utf-8").close()

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
        schedule, twt, completion = simulate(jobs, meta, rule, wc_units, wc_workers, worker_info, day_absences)

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
    jobs, meta, wc_df, wc_avail, wc_units, wc_workers, worker_info = load_and_preprocess()
    print(f"  {len(jobs)} orders with "
          f"{sum(len(v) for v in jobs.values())} operations loaded")

    results, best_rule = compare_all_rules(jobs, meta, wc_units, wc_workers, worker_info)

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
