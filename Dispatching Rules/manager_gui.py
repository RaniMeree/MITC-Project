"""
manager_gui.py
==============
Tkinter manager interface for weekly production scheduling.

Shows every employee (competence-level-2) in a Mon–Fri absence grid.
The manager ticks a box when an employee will not come in that day.
Clicking "Run Schedule" generates a weekly plan and shows the results
in a scrollable panel; "View Gantt" then opens the chart.

Run:
    python manager_gui.py
"""

import os
import sys
import io
import datetime
import queue
import threading
import importlib
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

import math

import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── patch dispatching_rules to use Demo / Data directory ─────────────────────
import dispatching_rules as dr

_HERE       = os.path.dirname(__file__)
_DATA_DIR   = os.path.join(_HERE, "Data")
dr.DATA_DIR = _DATA_DIR          # same override as run_demo_dispatching.py

# ── current work-week (Mon – Fri) ────────────────────────────────────────────
_today   = datetime.date.today()
_monday  = _today - datetime.timedelta(days=_today.weekday())
WEEK        = [_monday + datetime.timedelta(days=i) for i in range(5)]
DAY_NAMES   = ["Mon", "Tue", "Wed", "Thu", "Fri"]
WEEK_LABEL  = f"{WEEK[0].strftime('%b %d')} – {WEEK[4].strftime('%b %d, %Y')}"

# ── demo constants ────────────────────────────────────────────────────────────
ORDER_NAMES   = {1001: "Order-A", 1002: "Order-B", 1003: "Order-C"}
ORDER_COLOURS = {1001: "#4C72B0", 1002: "#DD8452", 1003: "#55A868"}


def _make_tooltip(widget, text):
    """Show a small popup with `text` when hovering over `widget`."""
    tip = None

    def _show(event):
        nonlocal tip
        x = widget.winfo_rootx() + 10
        y = widget.winfo_rooty() + widget.winfo_height() + 4
        tip = tk.Toplevel(widget)
        tip.wm_overrideredirect(True)
        tip.wm_geometry(f"+{x}+{y}")
        tk.Label(tip, text=text, justify="left",
                 background="#ffffe0", relief="solid", borderwidth=1,
                 font=("Segoe UI", 8), padx=4, pady=3).pack()

    def _hide(event):
        nonlocal tip
        if tip:
            tip.destroy()
            tip = None

    widget.bind("<Enter>", _show)
    widget.bind("<Leave>", _hide)


# ══════════════════════════════════════════════════════════════════════════════
class ManagerApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title(f"Production Schedule Manager — Week of {WEEK_LABEL}")
        root.geometry("1280x820")
        root.configure(bg="#f0f2f5")
        root.minsize(900, 600)

        # runtime state — dispatching rules
        self.last_results  = None
        self.last_best     = None
        self.last_meta     = None
        self.last_wc_names = None
        self.last_base_dt  = None

        # runtime state — GA
        self._last_jobs        = None
        self._last_wc_units    = None
        self._last_wc_workers  = None
        self._last_worker_info = None
        self._last_day_abs     = None
        self._ga_queue         = queue.Queue()
        self._ga_running       = False
        self._gantt_rule_adder = None
        self._ga_btn           = None

        self._build_ui()
        self._load_worker_data()

    # ──────────────────────────────────────────────────────────────────────────
    # UI CONSTRUCTION
    # ──────────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── header strip ──────────────────────────────────────────────────────
        hdr = tk.Frame(self.root, bg="#1a252f", pady=10)
        hdr.pack(fill="x")
        tk.Label(hdr, text="Production Schedule Manager",
                 font=("Segoe UI", 17, "bold"), fg="white",
                 bg="#1a252f").pack()
        tk.Label(hdr, text=f"Scheduling Week: {WEEK_LABEL}",
                 font=("Segoe UI", 10), fg="#95a5a6",
                 bg="#1a252f").pack()

        # ── two-column body ───────────────────────────────────────────────────
        body = tk.PanedWindow(self.root, orient="horizontal",
                              bg="#f0f2f5", sashwidth=6, sashrelief="flat")
        body.pack(fill="both", expand=True, padx=10, pady=8)

        # LEFT: absence grid
        left = tk.Frame(body, bg="#f0f2f5")
        body.add(left, minsize=340)
        self._build_left(left)

        # RIGHT: results
        right = tk.Frame(body, bg="#f0f2f5")
        body.add(right, minsize=500)
        self._build_right(right)

    def _build_left(self, parent):
        # section title
        tk.Label(parent, text="Employee Availability",
                 font=("Segoe UI", 12, "bold"), bg="#f0f2f5",
                 anchor="w").pack(fill="x")
        tk.Label(parent,
                 text=" Check a box = employee is ABSENT that day",
                 font=("Segoe UI", 9), fg="#e74c3c",
                 bg="#f0f2f5", anchor="w").pack(fill="x", pady=(0, 6))

        # scrollable canvas for the grid
        grid_outer = tk.Frame(parent, bg="#f0f2f5")
        grid_outer.pack(fill="both", expand=True)

        self._grid_canvas = tk.Canvas(grid_outer, bg="white",
                                      highlightthickness=1,
                                      highlightbackground="#c8c8c8")
        vsb = ttk.Scrollbar(grid_outer, orient="vertical",
                            command=self._grid_canvas.yview)
        self._grid_canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self._grid_canvas.pack(side="left", fill="both", expand=True)

        self._grid_inner = tk.Frame(self._grid_canvas, bg="white")
        self._cwin = self._grid_canvas.create_window(
            (0, 0), window=self._grid_inner, anchor="nw")

        self._grid_inner.bind("<Configure>",
            lambda e: self._grid_canvas.configure(
                scrollregion=self._grid_canvas.bbox("all")))
        self._grid_canvas.bind("<Configure>",
            lambda e: self._grid_canvas.itemconfig(self._cwin, width=e.width))

        # bind mousewheel for smooth scrolling
        self._grid_canvas.bind_all("<MouseWheel>",
            lambda e: self._grid_canvas.yview_scroll(-1*(e.delta//120), "units"))

        # buttons row
        btn_row = tk.Frame(parent, bg="#f0f2f5", pady=6)
        btn_row.pack(fill="x")

        tk.Button(btn_row, text="▶  Run Schedule",
                  command=self._run_schedule,
                  font=("Segoe UI", 11, "bold"),
                  bg="#27ae60", fg="white", activebackground="#1e8449",
                  relief="flat", padx=14, pady=7,
                  cursor="hand2").pack(side="left")

        tk.Button(btn_row, text="Clear All",
                  command=self._clear_all,
                  font=("Segoe UI", 10),
                  bg="#c0392b", fg="white", activebackground="#922b21",
                  relief="flat", padx=10, pady=7,
                  cursor="hand2").pack(side="left", padx=(8, 0))

        self._ga_btn = tk.Button(
                  btn_row, text="🧬  Run GA",
                  command=self._run_ga,
                  state="disabled",
                  font=("Segoe UI", 10),
                  bg="#8e44ad", fg="white", activebackground="#6c3483",
                  relief="flat", padx=10, pady=7,
                  cursor="hand2")
        # GA runs automatically — button hidden but kept for manual re-runs
        # self._ga_btn.pack(side="left", padx=(8, 0))

        # status bar
        self._status = tk.StringVar(value="Waiting…")
        tk.Label(parent, textvariable=self._status,
                 font=("Segoe UI", 9), fg="#555",
                 bg="#f0f2f5", anchor="w").pack(fill="x")

    def _build_right(self, parent):
        top = tk.Frame(parent, bg="#f0f2f5")
        top.pack(fill="x", pady=(0, 4))

        tk.Label(top, text="Schedule Results",
                 font=("Segoe UI", 12, "bold"),
                 bg="#f0f2f5").pack(side="left")

        tk.Button(top, text="📊  View Gantt Chart",
                  command=self._show_gantt,
                  font=("Segoe UI", 10),
                  bg="#2980b9", fg="white", activebackground="#1a5276",
                  relief="flat", padx=10, pady=5,
                  cursor="hand2").pack(side="right")

        self._result_text = scrolledtext.ScrolledText(
            parent, wrap="none",
            font=("Courier New", 9),
            state="disabled",
            bg="white", fg="#1a1a1a",
            relief="solid", borderwidth=1)
        self._result_text.pack(fill="both", expand=True)

    # ──────────────────────────────────────────────────────────────────────────
    # DATA LOADING  (worker grid)
    # ──────────────────────────────────────────────────────────────────────────
    def _load_worker_data(self):
        try:
            # wide format: id,WC_num1,WC_num2,... (comma-separated)
            # try comma first, fall back to semicolon
            comp_path = os.path.join(dr.DATA_DIR, "WorkerCompetences.csv")
            wide = pd.read_csv(comp_path, sep=",", dtype=str)
            if len(wide.columns) <= 1:
                wide = pd.read_csv(comp_path, sep=";", dtype=str)
            wide = wide.rename(columns={wide.columns[0]: "WorkerId"})
            wide["WorkerId"] = wide["WorkerId"].astype(str).str.strip()

            wc_cols = [c for c in wide.columns if c != "WorkerId"]
            comp_df = wide.melt(id_vars="WorkerId", value_vars=wc_cols,
                                var_name="WCNumber", value_name="CompetenceLevel")
            comp_df["CompetenceLevel"] = pd.to_numeric(
                comp_df["CompetenceLevel"], errors="coerce").fillna(0).astype(int)
            comp_df["WCNumber"] = pd.to_numeric(comp_df["WCNumber"], errors="coerce")

            wc_df = pd.read_csv(os.path.join(dr.DATA_DIR, "WorkCenters.tsv"), sep="\t")
            num_to_name = dict(zip(wc_df["Number"].astype(int), wc_df["Description"]))

            comp2 = comp_df[comp_df["CompetenceLevel"] == 2].copy()
            comp2["wc_name"] = comp2["WCNumber"].apply(
                lambda x: num_to_name.get(int(x), str(int(x))) if pd.notna(x) else "?")

            # group ALL competence-2 stations per worker (one row per worker)
            # absence is worker-level, not station-level
            from collections import defaultdict
            wc_per_worker = defaultdict(list)
            for _, row in comp2.sort_values(["WorkerId", "WCNumber"]).iterrows():
                wid = str(row["WorkerId"])
                wc_per_worker[wid].append(row["wc_name"])

            self.workers       = []
            self.absence_vars  = {}   # {worker_id_str: [BooleanVar × 5]}
            for wid in sorted(wc_per_worker, key=lambda x: int(x)):
                stations = wc_per_worker[wid]
                self.workers.append({
                    "id":    wid,
                    "label": f"Worker {wid}",
                    "wc":    ", ".join(stations),   # e.g. "GRIND, PRESS"
                    "multi": len(stations) > 1,
                })
                self.absence_vars[wid] = [tk.BooleanVar() for _ in range(5)]

            self._build_worker_grid()
            self._status.set(
                f"Loaded {len(self.workers)} workers with competence level 2")
        except Exception as exc:
            messagebox.showerror("Load Error", str(exc))

    def _build_worker_grid(self):
        g = self._grid_inner
        HDR_BG = "#1a252f"
        HDR_FG = "white"
        ROW_ODD  = "white"
        ROW_EVEN = "#f8f9fa"

        # ── header row ────────────────────────────────────────────────────────
        def _hcell(text, col, wide=18):
            tk.Label(g, text=text, font=("Segoe UI", 9, "bold"),
                     bg=HDR_BG, fg=HDR_FG,
                     width=wide, anchor="center",
                     padx=4, pady=6).grid(
                row=0, column=col, sticky="nsew", padx=1, pady=1)

        _hcell("Worker", 0, 10)
        _hcell("Work Centres", 1, 14)
        for c, (dname, ddate) in enumerate(zip(DAY_NAMES, WEEK)):
            _hcell(f"{dname}\n{ddate.strftime('%b %d')}", c + 2, 7)

        # ── worker rows ───────────────────────────────────────────────────────
        for r, w in enumerate(self.workers):
            bg  = ROW_ODD if r % 2 == 0 else ROW_EVEN
            row = r + 1

            tk.Label(g, text=w["label"], font=("Segoe UI", 9),
                     bg=bg, padx=6, anchor="w").grid(
                row=row, column=0, sticky="nsew", padx=1, pady=1)

            # Show compact count; full list in tooltip on hover
            stations  = [s.strip() for s in w["wc"].split(",")]
            n         = len(stations)
            short     = ", ".join(stations[:3]) + (f" +{n-3} more" if n > 3 else "")
            wc_fg     = "#1a6a9a" if w.get("multi") else "#555"
            wc_font   = ("Segoe UI", 9, "italic") if w.get("multi") else ("Segoe UI", 9)
            lbl = tk.Label(g, text=short, font=wc_font,
                           bg=bg, padx=4, anchor="w", fg=wc_fg)
            lbl.grid(row=row, column=1, sticky="nsew", padx=1, pady=1)

            # tooltip: full list shown in a small popup on hover
            _make_tooltip(lbl, "\n".join(stations))

            for c in range(5):
                tk.Checkbutton(
                    g,
                    variable=self.absence_vars[w["id"]][c],
                    bg=bg,
                    activebackground=bg,
                    selectcolor="#fde8e8",
                    cursor="hand2",
                ).grid(row=row, column=c + 2, padx=1, pady=1)

        # give the WC column more weight so it uses available space
        g.columnconfigure(0, weight=1)
        g.columnconfigure(1, weight=3)
        for col in range(2, 7):
            g.columnconfigure(col, weight=1)

    # ──────────────────────────────────────────────────────────────────────────
    # ACTIONS
    # ──────────────────────────────────────────────────────────────────────────
    def _clear_all(self):
        for w in self.workers:
            for var in self.absence_vars[w["id"]]:
                var.set(False)

    def _collect_absences(self):
        """
        Returns:
          absent_all_week  : set of worker IDs absent every Mon-Fri (→ excluded
                             from wc_workers entirely — machine may be blocked)
          day_absences     : {worker_id: set(0..4)} for partial absences
        """
        day_absences = {}
        for w in self.workers:
            days = {d for d in range(5)
                    if self.absence_vars[w["id"]][d].get()}
            if days:
                day_absences[w["id"]] = days

        absent_all_week   = {wid for wid, days in day_absences.items()
                             if len(days) == 5}
        partial_absences  = {wid: days for wid, days in day_absences.items()
                             if wid not in absent_all_week}
        return absent_all_week, partial_absences

    def _run_schedule(self):
        self._status.set("Generating schedule…")
        self.root.update()

        old_stdout = sys.stdout
        # sys.stdout = io.StringIO()          # suppress compare_all_rules prints  (disabled so VERBOSE prints reach terminal)

        try:
            absent_all_week, partial_absences = self._collect_absences()

            jobs, meta, wc_df, wc_avail, wc_units, wc_workers, worker_info = \
                dr.load_and_preprocess(absent_all_week)

            wc_names = dict(zip(wc_df["Id"], wc_df["Description"]))
            self.last_meta     = meta
            self.last_wc_names = wc_names

            # save for GA
            self._last_jobs        = jobs
            self._last_wc_units    = wc_units
            self._last_wc_workers  = wc_workers
            self._last_worker_info = worker_info
            self._last_day_abs     = partial_absences

            # Monday of the current work-week as the schedule base date
            base_dt = pd.Timestamp.now().normalize()
            base_dt = base_dt - pd.Timedelta(days=base_dt.dayofweek)  # Monday
            self.last_base_dt = base_dt

            results, best_rule = dr.compare_all_rules(
                jobs, meta, wc_units, wc_workers, worker_info,
                day_absences=partial_absences
            )

            self.last_results = results
            self.last_best    = best_rule
            if self._ga_btn:
                self._ga_btn.config(state="normal")

            # ── format results text ───────────────────────────────────────────
            buf = io.StringIO()

            # absence summary
            all_abs = {**{w: set(range(5)) for w in absent_all_week},
                       **partial_absences}
            if all_abs:
                buf.write("ABSENCE SUMMARY FOR THE WEEK\n")
                buf.write("=" * 68 + "\n")
                for wid in sorted(all_abs, key=lambda x: int(x)):
                    days_str = ", ".join(
                        f"{DAY_NAMES[d]} {WEEK[d].strftime('%b %d')}"
                        for d in sorted(all_abs[wid]))
                    buf.write(f"  Worker {wid:<4}  absent: {days_str}\n")
                buf.write("\n")

            def fmt(h):
                dt = base_dt + pd.Timedelta(hours=h)
                return dt.strftime("%a %b %d %H:%M")

            for rule in dr.RULES:
                buf.write("=" * 75 + "\n")
                best_tag = "  <-- BEST" if rule == best_rule else ""
                buf.write(f"  RULE: {rule}  —  TWT={results[rule]['twt']:.1f}h"
                          f"  late={results[rule]['late']}{best_tag}\n")
                buf.write("=" * 75 + "\n\n")

                sched = sorted(results[rule]["schedule"],
                               key=lambda x: x["start"])
                comp  = results[rule]["completion"]

                # operation table
                buf.write(f"  {'#':>3}  {'Machine':<8}  {'Order':<10}  "
                          f"{'Op':>4}  {'Worker':<8}  "
                          f"{'Start':<18}  {'End':<18}  "
                          f"{'Prio':>5}  {'Status':>6}\n")
                buf.write(f"  {'-'*3}  {'-'*8}  {'-'*10}  "
                          f"{'-'*4}  {'-'*8}  "
                          f"{'-'*18}  {'-'*18}  "
                          f"{'-'*5}  {'-'*6}\n")

                for idx, entry in enumerate(sched, 1):
                    m_id   = entry["machine"]
                    oid    = entry["order_id"]
                    prio   = int(meta[oid]["priority"])
                    late   = "LATE" if entry["end"] > meta[oid]["due_h"] else "OK"
                    mname  = wc_names.get(m_id, str(m_id))
                    olabel = ORDER_NAMES.get(oid, str(oid))
                    worker = str(entry.get("worker") or "-")
                    buf.write(
                        f"  {idx:>3}  {mname:<8}  {olabel:<10}  "
                        f"{int(entry['op_num']):>4}  {worker:<8}  "
                        f"{fmt(entry['start']):<18}  {fmt(entry['end']):<18}  "
                        f"{prio:>5}  {late:>6}\n")

                buf.write("\n  Order summary:\n")
                for oid in sorted(comp):
                    ct   = comp[oid]
                    due  = meta[oid]["due_h"]
                    prio = int(meta[oid]["priority"])
                    tard = max(0.0, ct - due)
                    stat = f"LATE by {tard:.1f}h" if tard > 0 else "ON TIME"
                    buf.write(
                        f"    {ORDER_NAMES.get(oid, str(oid)):<10}  "
                        f"priority={prio}  "
                        f"finished={fmt(ct)}  {stat}\n")
                buf.write("\n")

            # final comparison table
            buf.write("=" * 75 + "\n")
            buf.write("  FINAL COMPARISON\n")
            buf.write("=" * 75 + "\n")
            buf.write(f"  {'Rule':<6}  {'TWT (h)':>10}  {'Late':>6}  "
                      f"{'Makespan':>12}  {'Winner':>10}\n")
            buf.write(f"  {'-'*6}  {'-'*10}  {'-'*6}  {'-'*12}  {'-'*10}\n")
            for rule in dr.RULES:
                twt  = results[rule]["twt"]
                late = results[rule]["late"]
                c    = results[rule]["completion"]
                mksp = max(c.values()) if c else 0
                star = "<-- BEST" if rule == best_rule else ""
                buf.write(f"  {rule:<6}  {twt:>10.1f}  {late:>6}  "
                          f"{fmt(mksp):>12}  {star}\n")

            self._set_result_text(buf.getvalue())
            self._status.set(
                f"Done — Best rule: {best_rule}  "
                f"(TWT={results[best_rule]['twt']:.1f} h, "
                f"late={results[best_rule]['late']})")
                # f"late={results[best_rule]['late']})  |  Starting GA…")  # GA disabled

        except Exception as exc:
            messagebox.showerror("Schedule Error", str(exc))
            self._status.set(f"Error: {exc}")
        finally:
            sys.stdout = old_stdout

        # # automatically kick off GA optimisation in the background
        # self.root.after(100, self._run_ga)

    # ──────────────────────────────────────────────────────────────────────────
    # RESULT TEXT WIDGET
    # ──────────────────────────────────────────────────────────────────────────
    def _set_result_text(self, text: str):
        self._result_text.config(state="normal")
        self._result_text.delete("1.0", "end")
        self._result_text.insert("1.0", text)
        self._result_text.config(state="disabled")

    # ──────────────────────────────────────────────────────────────────────────
    # GENETIC ALGORITHM
    # ──────────────────────────────────────────────────────────────────────────
    def _run_ga(self):
        """Start GA optimisation in a background thread."""
        if self._last_jobs is None:
            messagebox.showinfo("No Data", "Run the dispatching rules first.")
            return
        if self._ga_running:
            return

        self._ga_running = True
        self._ga_btn.config(state="disabled", text="⏳  GA running…")
        self._status.set("GA starting…")
        self.root.update()

        # flush any leftover messages from a previous run
        while not self._ga_queue.empty():
            try:
                self._ga_queue.get_nowait()
            except queue.Empty:
                break

        def _worker():
            try:
                _ga_dir = os.path.join(_HERE, "..", "Solution2 - GA-1")
                if _ga_dir not in sys.path:
                    sys.path.insert(0, _ga_dir)
                import ga_solver
                importlib.reload(ga_solver)

                def _progress(gen, twt):
                    self._ga_queue.put(("progress", gen, twt))

                sched, twt, comp, history = ga_solver.ga_solve(
                    self._last_jobs,
                    self.last_meta,
                    wc_units     = self._last_wc_units,
                    wc_workers   = self._last_wc_workers,
                    worker_info  = self._last_worker_info,
                    day_absences = self._last_day_abs,
                    progress_callback = _progress,
                )
                self._ga_queue.put(("done", sched, twt, comp, history))
            except Exception:
                import traceback
                self._ga_queue.put(("error", traceback.format_exc()))

        threading.Thread(target=_worker, daemon=True).start()
        self.root.after(400, self._poll_ga)

    def _poll_ga(self):
        """Called repeatedly via after() to check the GA background thread.
        Drains ALL pending messages in one call so the UI never lags behind."""
        last_progress = None
        finished_msg  = None

        # drain the entire queue in one shot
        while True:
            try:
                msg = self._ga_queue.get_nowait()
            except queue.Empty:
                break
            if msg[0] == "progress":
                last_progress = msg          # keep only the latest progress
            else:
                finished_msg = msg           # "done" or "error"
                break                        # stop draining — terminal message

        if last_progress:
            gen, twt = last_progress[1], last_progress[2]
            self._status.set(f"GA  gen {gen}  ·  best TWT = {twt:.1f} h")

        if finished_msg is None:
            # not finished yet — check again soon
            self.root.after(200, self._poll_ga)
            return

        # use finished_msg where the original code expected msg
        msg  = finished_msg
        kind = msg[0]

        if kind == "done":
            sched, twt, comp, history = msg[1], msg[2], msg[3], msg[4]
            late = sum(1 for oid, ct in comp.items()
                       if ct > self.last_meta[oid]["due_h"])
            self.last_results["GA"] = {
                "twt":        twt,
                "late":       late,
                "schedule":   sched,
                "completion": comp,
                "history":    history,
            }
            self._ga_running = False
            self._ga_btn.config(state="normal", text="🧬  Run GA")
            best_rule_twt = self.last_results[self.last_best]["twt"]
            better = twt < best_rule_twt
            self._status.set(
                f"GA done!  TWT = {twt:.1f} h,  late = {late}"
                + ("  ← BETTER than all rules!" if better
                   else f"  (best rule {self.last_best} = {best_rule_twt:.1f} h)"))
            if self._gantt_rule_adder:
                try:
                    self._gantt_rule_adder("GA", better)
                except Exception:
                    pass
            return

        if kind == "error":
            self._ga_running = False
            self._ga_btn.config(state="normal", text="🧬  Run GA")
            messagebox.showerror("GA Error", msg[1])
            self._status.set("GA failed.")

    # ──────────────────────────────────────────────────────────────────────────
    # GANTT CHART
    # ──────────────────────────────────────────────────────────────────────────
    def _show_gantt(self):
        if self.last_results is None:
            messagebox.showinfo("No Results",
                                "Run the schedule first, then view the chart.")
            return

        results   = self.last_results
        best_rule = self.last_best
        meta      = self.last_meta
        wc_names  = self.last_wc_names
        base_dt   = self.last_base_dt

        # ── open embedded Gantt window ────────────────────────────────────────
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        win = tk.Toplevel(self.root)
        win.title("Gantt Chart")
        win.geometry("1400x750")
        win.configure(bg="#1a252f")

        # rule selector toolbar
        toolbar = tk.Frame(win, bg="#1a252f", pady=6)
        toolbar.pack(fill="x", padx=10)

        tk.Label(toolbar, text="Select:", font=("Segoe UI", 10, "bold"),
                 fg="white", bg="#1a252f").pack(side="left", padx=(4, 8))

        selected_rule = tk.StringVar(value=best_rule)
        rule_buttons  = {}

        def _overall_best():
            """Key in results with the lowest TWT (may be GA)."""
            return min(results, key=lambda r: results[r]["twt"])

        def _refresh_buttons():
            ob = _overall_best()
            for rule, btn in rule_buttons.items():
                is_sel  = (rule == selected_rule.get())
                is_best = (rule == ob)
                if is_sel:
                    bg, fg = "#2980b9", "white"
                elif is_best:
                    bg, fg = "#27ae60", "white"
                else:
                    bg, fg = "#2c3e50", "#bdc3c7"
                btn.config(bg=bg, fg=fg)

        def _on_rule_select(rule):
            selected_rule.set(rule)
            _refresh_buttons()
            _redraw(rule)

        def _make_btn(rule, is_ga=False):
            ob       = _overall_best()
            is_best  = (rule == ob)
            if is_ga:
                lbl = "GA  🧬★" if is_best else "GA  🧬"
            else:
                lbl = f"{rule}  ★" if is_best else rule
            btn = tk.Button(
                toolbar, text=lbl,
                font=("Segoe UI", 10, "bold" if is_best else "normal"),
                bg="#8e44ad" if is_ga else "#2c3e50",
                fg="white",
                relief="flat", padx=14, pady=5, cursor="hand2",
                command=lambda r=rule: _on_rule_select(r))
            btn.pack(side="left", padx=3)
            rule_buttons[rule] = btn

        # create buttons for all dispatching rules
        for rule in dr.RULES:
            _make_btn(rule)

        # # if GA results already available (e.g. window reopened) add GA button
        # if "GA" in results:
        #     _make_btn("GA", is_ga=True)

        # stats label (right side)
        stats_var = tk.StringVar()
        tk.Label(toolbar, textvariable=stats_var,
                 font=("Segoe UI", 9), fg="#95a5a6",
                 bg="#1a252f").pack(side="right", padx=8)

        _refresh_buttons()

        # chart area
        chart_frame = tk.Frame(win, bg="#ffffff")
        chart_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        fig, ax_holder = plt.subplots(1, 1, figsize=(22, 7))
        fig.patch.set_facecolor("#ffffff")
        canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        canvas.get_tk_widget().pack(fill="both", expand=True)

        def _redraw(rule):
            ax_holder.cla()
            self._render_gantt(ax_holder, rule, results, meta, wc_names, base_dt)
            twt   = results[rule]["twt"]
            late  = results[rule]["late"]
            ob    = _overall_best()
            tag   = "  ★ BEST" if rule == ob else ""
            label = "Genetic Algorithm (GA)" if rule == "GA" else f"Rule: {rule}"
            stats_var.set(
                f"{label}{tag}   |   TWT = {twt:.1f} h   |   Late orders = {late}")
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            canvas.draw()

        # # callback so _poll_ga can add the GA button while this window is open
        # def _add_ga_btn(rule, is_better):
        #     if rule not in rule_buttons:
        #         _make_btn(rule, is_ga=True)
        #     _refresh_buttons()
        # self._gantt_rule_adder = _add_ga_btn

        def _on_close():
            # self._gantt_rule_adder = None
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", _on_close)

        _redraw(best_rule)
        return  # rendering logic lives in _render_gantt

    def _render_gantt(self, ax, rule, results, meta, wc_names, base_dt):
        """Draw the Gantt for *rule* onto *ax*."""
        schedule = results[rule]["schedule"]

        # ── working-hours constants ────────────────────────────────────────────
        SHIFT_START = 8.0          # 08:00
        SHIFT_END   = 18.0         # 18:00
        SHIFT_DUR   = SHIFT_END - SHIFT_START   # 10 working hours per day
        N_DAYS      = 5            # Mon – Fri only

        def ceil15(h):
            """Round simulation hours UP to the nearest 15-minute boundary."""
            return math.ceil(h * 4) / 4

        def to_x(sim_h):
            """Simulation hours (Mon 00:00 = 0) → compressed x coordinate.
            Each working day occupies SHIFT_DUR (10) units.
            Hours outside 08:00-18:00 are clamped to the shift boundaries.
            """
            day = int(sim_h // 24)
            hod = sim_h % 24
            hod = max(SHIFT_START, min(SHIFT_END, hod))
            return day * SHIFT_DUR + (hod - SHIFT_START)

        # ── machine layout — order of first appearance ─────────────────────────
        machine_order, seen = [], set()
        for entry in sorted(schedule, key=lambda x: x["start"]):
            m = entry["machine"]
            if m not in seen:
                machine_order.append(m)
                seen.add(m)
        machine_y  = {m: i for i, m in enumerate(machine_order)}
        n_machines = len(machine_order)

        BAR_H       = 0.55
        total_width = N_DAYS * SHIFT_DUR

        ax.set_xlim(0, total_width)
        ax.set_ylim(-0.7, n_machines - 0.3)
        ax.invert_yaxis()

        # ── alternating day backgrounds ───────────────────────────────────────
        day_cols = ["#f5f8ff", "#edf2fb"]
        for d in range(N_DAYS):
            x0 = d * SHIFT_DUR
            ax.axvspan(x0, x0 + SHIFT_DUR,
                       color=day_cols[d % 2], alpha=1.0, zorder=0)
            ax.axvline(x0, color="#c0c8d8", linewidth=1.2, zorder=1)
        ax.axvline(total_width, color="#c0c8d8", linewidth=1.2, zorder=1)

        # ── 15-min vertical gridlines  (solid at full hours, dotted otherwise) ─
        for d in range(N_DAYS):
            for slot in range(int(SHIFT_DUR * 4) + 1):   # 0 … 40
                x       = d * SHIFT_DUR + slot * 0.25
                is_hour = (slot % 4 == 0)
                ax.axvline(x,
                           color="#b0b8cc",
                           linewidth=0.8 if is_hour else 0.3,
                           linestyle="-" if is_hour else ":",
                           alpha=0.6 if is_hour else 0.3,
                           zorder=1)

        # ── operation bars ────────────────────────────────────────────────────
        for entry in schedule:
            m       = entry["machine"]
            oid     = entry["order_id"]
            y       = machine_y[m]
            start_x = to_x(ceil15(entry["start"]))
            end_x   = to_x(ceil15(entry["end"]))
            dur_x   = max(0.25, end_x - start_x)   # minimum one 15-min slot
            color   = ORDER_COLOURS.get(oid, "#999")
            late    = entry["end"] > meta[oid]["due_h"]

            ax.barh(y, dur_x, left=start_x, height=BAR_H,
                    color=color, edgecolor="white",
                    linewidth=0.8, alpha=0.9, zorder=3)
            if dur_x >= 0.5:
                lbl = (f"{ORDER_NAMES.get(oid, str(oid)).split('-')[1]}"
                       f"-op{int(entry['op_num'])}")
                ax.text(start_x + dur_x / 2, y, lbl,
                        ha="center", va="center",
                        fontsize=7.5, color="white", fontweight="bold", zorder=4)
            if late:
                ax.barh(y, dur_x, left=start_x, height=BAR_H,
                        color="none", edgecolor="red", linewidth=2.0, zorder=4)

        # ── due-date markers ──────────────────────────────────────────────────
        for oid, m_info in meta.items():
            due_x = to_x(m_info["due_h"])
            if 0 <= due_x <= total_width:
                color = ORDER_COLOURS.get(oid, "#999")
                ax.axvline(due_x, color=color, linestyle="--",
                           linewidth=1.2, alpha=0.7, zorder=2)
                # y=-0.6 is above machine 0 in inverted-y space → top of chart
                ax.text(due_x, -0.6,
                        f"{ORDER_NAMES.get(oid, str(oid))}\ndue",
                        ha="center", va="bottom",
                        fontsize=7, color=color, zorder=5)

        # ── bottom x-axis: week number + day + date ───────────────────────────
        day_ticks  = [(d + 0.5) * SHIFT_DUR for d in range(N_DAYS)]
        day_labels = []
        for d in range(N_DAYS):
            dt_day = (base_dt + pd.Timedelta(days=d)).date()
            wn     = dt_day.isocalendar()[1]
            day_labels.append(f"W{wn}  —  {dt_day.strftime('%A, %b %d')}")
        ax.set_xticks(day_ticks)
        ax.set_xticklabels(day_labels, fontsize=9)
        ax.tick_params(axis="x", which="major", length=0)
        ax.set_xlabel("")

        # ── top x-axis: 08:00 → 08:15 → … → 18:00  (repeats each day) ────────
        ax2 = ax.twiny()
        ax2.set_xlim(0, total_width)

        t_ticks  = []
        t_labels = []
        for d in range(N_DAYS):
            for slot in range(int(SHIFT_DUR * 4) + 1):   # 0 … 40
                x      = d * SHIFT_DUR + slot * 0.25
                h_abs  = SHIFT_START + slot * 0.25
                hour   = int(h_abs)
                minute = round((h_abs - hour) * 60)
                t_ticks.append(x)
                # label every full hour; blank string → tick mark only
                t_labels.append(f"{hour:02d}:00" if minute == 0 else "")

        ax2.set_xticks(t_ticks)
        ax2.set_xticklabels(t_labels, fontsize=7, rotation=90)
        ax2.tick_params(axis="x", which="major", length=4, direction="out")
        ax2.set_xlabel("Working Hours (08:00 – 18:00)", fontsize=9, labelpad=6)

        # ── y-axis labels ─────────────────────────────────────────────────────
        ax.set_yticks(range(n_machines))
        ax.set_yticklabels(
            [wc_names.get(m, str(m)) for m in machine_order], fontsize=10)

        twt  = results[rule]["twt"]
        late = results[rule]["late"]
        rule_label = "Genetic Algorithm (GA)" if rule == "GA" else f"Rule: {rule}"
        ax.set_title(
            f"Gantt Chart — {rule_label}  "
            f"(TWT = {twt:.1f} h,  late orders = {late})",
            fontsize=12, fontweight="bold", pad=50)

        patches = [
            mpatches.Patch(
                color=ORDER_COLOURS[oid],
                label=f"{ORDER_NAMES[oid]}  (priority = {int(meta[oid]['priority'])})")
            for oid in sorted(ORDER_COLOURS) if oid in meta
        ]
        patches.append(mpatches.Patch(
            facecolor="none", edgecolor="red", linewidth=2, label="LATE finish"))
        ax.legend(handles=patches, loc="lower right", fontsize=9)


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    root = tk.Tk()
    ManagerApp(root)
    root.mainloop()
