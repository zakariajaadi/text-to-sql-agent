# offline/studio/jobs.py

"""
The pipeline steps the studio runs: the CLI of offline/cli.py in a subprocess, exactly what the
Makefile runs, one job at a time (a Hive or Impala job at a time, and one writer per YAML file).
The studio process stays light (no model, no Chroma) and a job can be stopped.

The output is kept line by line, its colors stripped; the lines of the CLI give the progress
("Profiling: <table>", "Not profiled: <table>").
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
STARTED  = re.compile(r"\] Profiling: (?P<table>\S+)$")
FAILED   = re.compile(r"\] Not profiled: (?P<table>\S+) ")
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
        if started := STARTED.search(message):
            if self.current and self.current not in self.failed:
                self.done.append(self.current)
            self.current, self.current_since = started["table"], time.time()
        elif failed := FAILED.search(message):
            self.failed.append(failed["table"])
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
