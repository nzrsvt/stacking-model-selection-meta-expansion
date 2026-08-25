import ctypes
import io
import os
import queue
import sys
import threading
import tkinter as tk
from tkinter import filedialog, font, ttk

class PipelineCancelled(BaseException):
    """Raised inside the worker thread when the user clicks Cancel."""

class QueueWriter(io.TextIOBase):
    def __init__(self, q: queue.Queue):
        self._q = q

    def write(self, text):
        if text:
            self._q.put(("log", text))
        return len(text)

    def flush(self):
        pass


def parse_results(log_text: str) -> list[tuple[str, str]]:
    rows = []
    for line in log_text.splitlines():
        line = line.strip()
        markers = [
            "Відібрані моделі",
            "Обраний n_parts",
            "Межі інтервалів",
            "rho_pred_max",
            "rho_err_bad",
            "med_norm_min",
            "div_max",
        ]
        for m in ["MAPE", "RMSE", "MAE", "Max Error", "Median AE", "MSE", "R2"]:
            markers.append(f"Test {m} БЕЗ розширення")
            markers.append(f"Test {m} кусково-лінійне")
            
        for marker in markers:
            if marker in line and ":" in line:
                parts = line.split(":", 1)
                label = parts[0].strip().lstrip(":").strip()
                value = parts[1].strip()
                rows.append((label, value))
                break
    return rows

class PipelineGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Stacking Ensemble Pipeline")
        self.geometry("1100x750")
        self.minsize(700, 560)
        self.configure(bg="#f3f4f6")

        self._setup_styles()

        self._q: queue.Queue = queue.Queue()
        self._running = False
        self._log_buf: list[str] = []
        self._worker_thread: threading.Thread | None = None

        self._build_ui()
        self._poll_queue()

    def _setup_styles(self):
        style = ttk.Style(self)
        try:
            style.theme_use('clam')
        except tk.TclError:
            pass
        
        style.configure(".", background="#f3f4f6", foreground="#1f2937")
        style.configure("TFrame", background="#f3f4f6")
        style.configure("White.TFrame", background="#ffffff")
        
        style.configure("TLabelframe", background="#ffffff", bordercolor="#d1d5db", relief="flat")
        style.configure("TLabelframe.Label", background="#ffffff", font=("", 10, "bold"), foreground="#374151", padding=(5, 0))
        
        style.configure("TLabel", background="#ffffff")
        style.configure("Transparent.TLabel", background="#f3f4f6")
        
        style.configure("TButton", font=("", 10, "bold"), padding=6, relief="flat", background="#e5e7eb")
        style.map("TButton", background=[("active", "#d1d5db")])
        
        style.configure("Primary.TButton", background="#2563eb", foreground="white")
        style.map("Primary.TButton", background=[("active", "#1d4ed8"), ("disabled", "#93c5fd")])
        
        style.configure("Danger.TButton", background="#dc2626", foreground="white")
        style.map("Danger.TButton", background=[("active", "#b91c1c"), ("disabled", "#fca5a5")])
        
        style.configure("Treeview", rowheight=28, bordercolor="#d1d5db", relief="flat")
        style.configure("Treeview.Heading", font=("", 10, "bold"), background="#f9fafb", foreground="#374151")
        style.map("Treeview", background=[("selected", "#dbeafe")], foreground=[("selected", "#1e40af")])


    def _build_ui(self):
        main_frame = ttk.Frame(self, padding=15)
        main_frame.pack(fill="both", expand=True)

        data_frame = ttk.LabelFrame(main_frame, text=" Data Source ", padding=(15, 10))
        data_frame.pack(fill="x", pady=(0, 15))
        data_frame.columnconfigure(1, weight=1)

        ttk.Label(data_frame, text="Dataset (.csv):").grid(row=0, column=0, sticky="w")
        self.path_var = tk.StringVar()
        ttk.Entry(data_frame, textvariable=self.path_var).grid(row=0, column=1, sticky="ew", padx=10)
        ttk.Button(data_frame, text="Browse…", command=self._browse).grid(row=0, column=2, sticky="e")

        opts_frame = ttk.LabelFrame(main_frame, text=" Options ", padding=(15, 8))
        opts_frame.pack(fill="x", pady=(0, 15))

        self.skip_selection_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            opts_frame,
            text="Skip base-model selection",
            variable=self.skip_selection_var,
        ).pack(side="left")

        ctrl_frame = ttk.Frame(main_frame, style="TFrame")
        ctrl_frame.pack(fill="x", pady=(0, 15))
        ctrl_frame.columnconfigure(1, weight=1)

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(ctrl_frame, textvariable=self.status_var, font=("", 10, "italic"), style="Transparent.TLabel").grid(row=0, column=0, sticky="w")
        
        self.progress = ttk.Progressbar(ctrl_frame, mode="indeterminate")
        self.progress.grid(row=0, column=1, sticky="ew", padx=20)

        self.run_btn = ttk.Button(ctrl_frame, text="▶  Run Pipeline", style="Primary.TButton", command=self._run)
        self.run_btn.grid(row=0, column=2, padx=(0, 10))

        self.cancel_btn = ttk.Button(ctrl_frame, text="⏹  Cancel", style="Danger.TButton", state="disabled", command=self._cancel)
        self.cancel_btn.grid(row=0, column=3)

        paned = tk.PanedWindow(main_frame, orient="horizontal", sashwidth=6, sashrelief="raised", bg="#e5e7eb")
        paned.pack(fill="both", expand=True)

        log_frame = ttk.LabelFrame(paned, text=" Pipeline Output ", padding=5)
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)

        mono = font.Font(family="Consolas", size=10)
        self.log_box = tk.Text(
            log_frame, wrap="none", state="disabled",
            font=mono, bg="#1e1e1e", fg="#d4d4d4",
            insertbackground="white", relief="flat", bd=0, highlightthickness=0
        )
        log_vsb = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_box.yview)
        log_hsb = ttk.Scrollbar(log_frame, orient="horizontal", command=self.log_box.xview)
        self.log_box.configure(yscrollcommand=log_vsb.set, xscrollcommand=log_hsb.set)

        self.log_box.grid(row=0, column=0, sticky="nsew")
        log_vsb.grid(row=0, column=1, sticky="ns")
        log_hsb.grid(row=1, column=0, sticky="ew")

        self.log_box.tag_config("sep",    foreground="#6a9955")
        self.log_box.tag_config("header", foreground="#4ec9b0", font=font.Font(family="Consolas", size=10, weight="bold"))
        self.log_box.tag_config("ok",     foreground="#b5cea8")
        self.log_box.tag_config("warn",   foreground="#ce9178")
        self.log_box.tag_config("err",    foreground="#f44747")
        self.log_box.tag_config("result", foreground="#dcdcaa")

        btn_row = ttk.Frame(log_frame, style="White.TFrame")
        btn_row.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 2))
        ttk.Button(btn_row, text="Clear", command=self._clear_log).pack(side="left")
        ttk.Button(btn_row, text="Copy all", command=self._copy_log).pack(side="left", padx=5)

        paned.add(log_frame, minsize=400, stretch="always")

        res_frame = ttk.LabelFrame(paned, text=" Results ", padding=5)
        res_frame.rowconfigure(0, weight=1)
        res_frame.columnconfigure(0, weight=1)

        self.res_tree = ttk.Treeview(
            res_frame, columns=("metric", "value"),
            show="headings", height=20,
        )
        self.res_tree.heading("metric", text="Metric")
        self.res_tree.heading("value",  text="Value")
        self.res_tree.column("metric", width=220, anchor="w")
        self.res_tree.column("value",  width=180, anchor="w")

        res_vsb = ttk.Scrollbar(res_frame, orient="vertical", command=self.res_tree.yview)
        res_hsb = ttk.Scrollbar(res_frame, orient="horizontal", command=self.res_tree.xview)
        self.res_tree.configure(yscrollcommand=res_vsb.set, xscrollcommand=res_hsb.set)

        self.res_tree.grid(row=0, column=0, sticky="nsew")
        res_vsb.grid(row=0, column=1, sticky="ns")
        res_hsb.grid(row=1, column=0, sticky="ew")

        paned.add(res_frame, minsize=250)

    def _browse(self):
        path = filedialog.askopenfilename(
            title="Select CSV dataset",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self.path_var.set(path)

    def _clear_log(self):
        self.log_box.config(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.config(state="disabled")
        self._log_buf.clear()

    def _copy_log(self):
        self.clipboard_clear()
        self.clipboard_append("".join(self._log_buf))

    def _run(self):
        csv_path = self.path_var.get().strip()
        if not csv_path:
            self._status("Please select a CSV file first.", error=True)
            return
        if not os.path.isfile(csv_path):
            self._status(f"File not found: {csv_path}", error=True)
            return
        if self._running:
            return

        self._clear_log()
        self._clear_results()
        self._running = True
        self.run_btn.config(state="disabled")
        self.cancel_btn.config(state="normal")
        self.progress.start(12)
        self._status("Running…")

        params = {
            "csv_file":       csv_path,
            "skip_selection": self.skip_selection_var.get(),
        }
        t = threading.Thread(target=self._worker, kwargs=params, daemon=True)
        self._worker_thread = t
        t.start()

    def _worker(self, **params):
        writer = QueueWriter(self._q)
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = writer
        sys.stderr = writer
        try:
            from pipeline import run_full_pipeline 
            run_full_pipeline(**params)
            self._q.put(("done", None))
        except PipelineCancelled:
            self._q.put(("cancelled", None))
        except Exception as exc:  
            import traceback
            self._q.put(("log", f"\n{'='*60}\nERROR: {exc}\n{traceback.format_exc()}\n"))
            self._q.put(("error", str(exc)))
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def _cancel(self):
        t = self._worker_thread
        if t is None or not t.is_alive():
            return
        tid = t.ident
        if tid is None:
            return
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_ulong(tid),
            ctypes.py_object(PipelineCancelled),
        )
        if res == 0:
            self._status("Cancel failed: thread not found.", error=True)
        elif res > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(tid), None)
            self._status("Cancel failed: unexpected thread state.", error=True)
        else:
            self._status("Cancelling…")
            self.cancel_btn.config(state="disabled")


    def _poll_queue(self):
        try:
            while True:
                kind, payload = self._q.get_nowait()
                if kind == "log":
                    self._append_log(payload)
                elif kind == "done":
                    self._on_done()
                elif kind == "cancelled":
                    self._on_cancelled()
                elif kind == "error":
                    self._on_error(payload)
        except queue.Empty:
            pass
        self.after(50, self._poll_queue)

    def _append_log(self, text: str):
        self._log_buf.append(text)
        self.log_box.config(state="normal")
        tag = self._classify(text)
        self.log_box.insert("end", text, tag)
        self.log_box.see("end")
        self.log_box.config(state="disabled")

    @staticmethod
    def _classify(text: str) -> str:
        t = text.strip()
        if t.startswith("=") or t.startswith("-"):
            return "sep"
        if any(k in t for k in ("КРОК", "ПАЙПЛАЙН", "ЗБЕРЕЖЕННЯ", "ПІДСУМОК",
                                  "АВТОКАЛІБРУВАННЯ", "ПОВНИЙ ЧАС")):
            return "header"
        if t.startswith("OK ") or "Рекомендовано" in t or "Відмінно" in t:
            return "ok"
        if t.startswith("X ") or "ERROR" in t or "Помилка" in t:
            return "err"
        if t.startswith("!") or "Обережно" in t:
            return "warn"
        if "Test " in t or "Train OOF " in t:
            return "result"
        return ""


    def _clear_results(self):
        for row in self.res_tree.get_children():
            self.res_tree.delete(row)

    def _populate_results(self):
        full_log = "".join(self._log_buf)
        rows = parse_results(full_log)
        self._clear_results()

        section = None
        sections = {
            "Відібрані":   "📦 Selection",
            "Обраний":     "📦 Selection",
            "Межі":        "📦 Selection",
            "Test ":       "📊 Metrics",
            "rho_pred":    "🔧 Thresholds",
            "rho_err":     "🔧 Thresholds",
            "med_norm":    "🔧 Thresholds",
            "div_max":     "🔧 Thresholds",
        }

        shown_sections: set = set()
        for label, value in rows:
            for key, sec in sections.items():
                if key in label:
                    section = sec
                    break

            if section and section not in shown_sections:
                self.res_tree.insert("", "end", values=(f"── {section} ──", ""), tags=("section",))
                shown_sections.add(section)

            self.res_tree.insert("", "end", values=(label, value))

        self.res_tree.tag_configure("section", foreground="#6b7280")

        full_log_lines = full_log.splitlines()
        for line in full_log_lines:
            if "-> " in line and ".xlsx" in line:
                fname = line.split("'")[1] if "'" in line else line.strip()
                self.res_tree.insert("", "end", values=("📄 Saved file", fname))

    def _on_done(self):
        self._running = False
        self._worker_thread = None
        self.run_btn.config(state="normal")
        self.cancel_btn.config(state="disabled")
        self.progress.stop()
        self._status("Done ✓")
        self._populate_results()

    def _on_cancelled(self):
        self._running = False
        self._worker_thread = None
        self.run_btn.config(state="normal")
        self.cancel_btn.config(state="disabled")
        self.progress.stop()
        self._append_log("\n⏹  Pipeline cancelled by user.\n")
        self._status("Cancelled.")

    def _on_error(self, msg: str):
        self._running = False
        self._worker_thread = None
        self.run_btn.config(state="normal")
        self.cancel_btn.config(state="disabled")
        self.progress.stop()
        self._status(f"Error: {msg}", error=True)

    def _status(self, text: str, error: bool = False):
        self.status_var.set(text)
        self.winfo_children()


if __name__ == "__main__":
    app = PipelineGUI()
    app.mainloop()