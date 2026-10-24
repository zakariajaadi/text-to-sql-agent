# offline/studio/jobs.py

"""
The pipeline steps the studio runs: the CLI of offline/cli.py in a subprocess, exactly what the
Makefile runs, one job at a time (a Hive or Impala job at a time, and one writer per YAML file).
The studio process stays light (no model, no Chroma) and a job can be stopped.

The output is kept line by line, its colors stripped; the lines of the CLI give the progress
("Profiling: <table>", "Not profiled: <table>", "Fetching values: <table>", "[3/23] <question>").
"""

import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field

from config import BASE_DIR

MAX_LINES  = 2000   # lines kept per job, the first ones dropped
STOP_GRACE = 30     # seconds a stopped job has to cancel its query (offline/query_cancel.py) before it is killed

ANSI     = re.compile(r"\x1b\[[0-9;]*m")
# A loguru line of config.py: "2026-10-23 10:12:03 | INFO     | module:function:12 - message"
LOG_LINE = re.compile(r"^\d{4}-\d\d-\d\d (?P<time>\d\d:\d\d:\d\d) \| (?P<level>\w+)\s*\| [^ ]+ - (?P<message>.*)$")
# A table started: profiled one at a time, described up to 8 at a time (offline/semantic_gen/tables.py)
STARTED  = re.compile(r"\] (?:Profiling|Describing): (?P<table>\S+)$")
FAILED   = re.compile(r"\] Not profiled: (?P<table>\S+) ")
CHECKING = "] Checking: "   # a relation measured on the data (offline/semantic_gen/relationships.py)
# The index (offline/indexer/schema_indexer.py): the values read table after table, then the documents
# built and written into Chroma, then the value index
FETCHING = re.compile(r"\] Fetching values: (?P<table>\S+) \(")
PHASES   = (("] Built: ", "documents"), (" schemas to Chroma.", "values"))
# A question of the eval asked to the agent (offline/eval/run.py)
QUESTION = re.compile(r"^\[(?P<n>\d+)/(?P<total>\d+)\] ")
# Lines of offline/query_cancel.py on a stop
CANCELLED     = "the running query is cancelled on the cluster"
NOT_CANCELLED = "the running query could not be cancelled"


@dataclass
class Job:
    id: int
    db: str
    step: str
    tables: list[str] | None
    command: list[str]
    started: float = field(default_factory=time.time)
    ended: float | None = None
    returncode: int | None = None
    stopped: bool = False
    stopped_at: float | None = None
    cancel: str | None = None   # on a stop, the running query: "cancelled" or "failed" on the cluster
    current: str | None = None
    current_since: float | None = None   # when the current table started: how long the database takes
    done: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    started_tables: list[str] = field(default_factory=list)   # every table started, in order
    checked: int = 0                                          # relations measured on the data
    phase: str | None = None                                  # index: "documents", then "values" (PHASES)
    progress: list[int] | None = None                         # eval: [question, questions]
    lines: list[dict] = field(default_factory=list)
    first_line: int = 0   # number of the first line kept

    @property
    def running(self) -> bool:
        return self.ended is None

    def add_line(self, raw: str) -> None:
        """Keep one line of the output and follow the progress it tells."""
        text = ANSI.sub("", raw).rstrip()
        if not text:
            return
        match = LOG_LINE.match(text)
        line  = match.groupdict() if match else {"time": None, "level": None, "message": text}
        message = line["message"]
        phase = next((name for marker, name in PHASES if marker in message), None)
        if started := STARTED.search(message) or FETCHING.search(message):
            self.started_tables.append(started["table"])
            if self.current and self.current not in self.failed:
                self.done.append(self.current)
            self.current, self.current_since = started["table"], time.time()
        elif phase:
            # The values are all read: the last table is done
            if self.current and self.current not in self.failed:
                self.done.append(self.current)
            self.current, self.current_since, self.phase = None, None, phase
        elif question := QUESTION.match(message):
            self.progress = [int(question["n"]), int(question["total"])]
        elif failed := FAILED.search(message):
            self.failed.append(failed["table"])
        elif CHECKING in message:
            self.checked += 1
        elif CANCELLED in message or NOT_CANCELLED in message:
            self.cancel = "cancelled" if CANCELLED in message else "failed"
        self.lines.append(line)
        if len(self.lines) > MAX_LINES:
            del self.lines[0]
            self.first_line += 1

    def finish(self, returncode: int) -> None:
        if returncode == 0 and self.current and self.current not in self.failed:
            self.done.append(self.current)
        self.current, self.current_since, self.returncode, self.ended = None, None, returncode, time.time()

    def to_dict(self, since: int = 0) -> dict:
        """
        The job, with its lines from number `since` on (the ones the page doesn't have yet).
        `now`: the clock of the server, for the page to count the seconds of current_since.
        """
        start = max(since - self.first_line, 0)
        return {
            "id": self.id, "db": self.db, "step": self.step, "tables": self.tables,
            "command": " ".join(self.command[1:]), "started": self.started, "ended": self.ended, "now": time.time(),
            "running": self.running, "returncode": self.returncode, "stopped": self.stopped,
            "stopped_at": self.stopped_at, "cancel": self.cancel,
            "current": self.current, "current_since": self.current_since, "done": self.done, "failed": self.failed,
            "started_tables": self.started_tables, "checked": self.checked, "phase": self.phase, "progress": self.progress,
            "lines": self.lines[start:], "next_line": self.first_line + len(self.lines),
        }


def profile_command(db: str, tables: list[str] | None, missing: bool) -> list[str]:
    """The CLI command of `make profile`: the tables asked for, or the ones missing from the profile."""
    command = [sys.executable, "-m", "offline.cli", "profile", "--db", db]
    if tables:
        command += ["--tables", ",".join(tables)]
    elif missing:
        command.append("--missing")
    return command


def index_command(db: str) -> list[str]:
    """The CLI command of `make index`: the documents of the tables and the value index, written again."""
    return [sys.executable, "-m", "offline.cli", "index", "--db", db]


def eval_command(db: str) -> list[str]:
    """The command of `make eval`: the questions of offline/eval/datasets/<db>.yaml asked to the agent."""
    return [sys.executable, "-m", "offline.eval.run", "--db", db]


def metrics_command(db: str) -> list[str]:
    """The CLI command of `make metrics`: metrics proposed by the LLM, added to metrics.yaml."""
    return [sys.executable, "-m", "offline.cli", "metrics", "--db", db]


def relations_command(db: str, check: bool) -> list[str]:
    """The CLI command of `make relations`: find the relations (measured and LLM), or only check them again."""
    command = [sys.executable, "-m", "offline.cli", "fk-relations", "--db", db]
    return command + ["--check"] if check else command


def semantics_command(db: str, tables: list[str] | None, fresh: bool) -> list[str]:
    """The CLI command of `make semantics`: the missing descriptions, or `tables` described again (fresh)."""
    command = [sys.executable, "-m", "offline.cli", "semantics", "--db", db]
    if tables:
        command += ["--tables", ",".join(tables)]
    if fresh:
        command.append("--fresh")
    return command


class JobRunner:
    """Runs one job at a time and keeps the last one, for the page to follow."""

    def __init__(self):
        self._lock    = threading.Lock()
        self._process: subprocess.Popen | None = None
        self.job: Job | None = None

    def start(self, db: str, step: str, command: list[str], tables: list[str] | None = None) -> Job:
        """Start a job; RuntimeError while another one runs."""
        with self._lock:
            if self.job and self.job.running:
                raise RuntimeError(f"A job is running: {self.job.step} on {self.job.db}")
            env = dict(os.environ, PYTHONUNBUFFERED="1",
                       PYTHONPATH=os.pathsep.join(filter(None, [str(BASE_DIR / "src"), os.getenv("PYTHONPATH")])))
            self.job = Job(id=(self.job.id + 1 if self.job else 1), db=db, step=step, tables=tables, command=command)
            self._process = subprocess.Popen(
                command, cwd=BASE_DIR, env=env, text=True, bufsize=1,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )
            threading.Thread(target=self._follow, args=(self.job, self._process), daemon=True).start()
            return self.job

    def stop(self) -> None:
        """
        Stop the running job, if any (profile.yaml keeps the tables measured before): SIGTERM, on which
        the CLI cancels its running query on the cluster and exits; killed if it is still there after
        STOP_GRACE seconds.
        """
        with self._lock:
            if self.job and self.job.running and self._process and not self.job.stopped:
                self.job.stopped, self.job.stopped_at = True, time.time()
                self._process.terminate()
                threading.Thread(target=self._kill_after_grace, args=(self._process,), daemon=True).start()

    @staticmethod
    def _kill_after_grace(process: subprocess.Popen) -> None:
        try:
            process.wait(STOP_GRACE)
        except subprocess.TimeoutExpired:
            process.kill()

    @staticmethod
    def _follow(job: Job, process: subprocess.Popen) -> None:
        for raw in process.stdout:
            job.add_line(raw)
        job.finish(process.wait())
