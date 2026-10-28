# offline/studio/jobs.py

"""
The pipeline steps the studio runs: the CLI of offline/cli.py in a subprocess, exactly what the
Makefile runs, one job at a time (a Hive or Impala job at a time, and one writer per YAML file).
The studio process stays light (no model, no Chroma) and a job can be stopped.

The output is kept line by line, its colors stripped; the lines of the CLI give the progress
("Profiling: <table>", "Not profiled: <table>", "Fetching values: <table>", "[3/23] <question>")
and what a job found (what the profile detected, the relations added).
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
# A table started: profiled (or its profile completed) one at a time, described up to 8 at a time
# (offline/semantic_gen/tables.py)
STARTED  = re.compile(r"\] (?:Profiling|Completing|Describing): (?P<table>\S+)$")
FAILED   = re.compile(r"\] Not (?:profiled|completed): (?P<table>\S+) ")
# What the profile detected in a table measured (offline/semantic_gen/profiling.py)
DETECTED = re.compile(r"\] Detected in \S+: (?P<formats>\d+) format\(s\), (?P<empty>\d+) empty column\(s\), "
                      r"(?P<personal>\d+) personal data column\(s\)$")
# A search of relations (offline/semantic_gen/relationships.py): the LLM asked, then the relations measured
# on the data one by one, then the new ones added
ASKING_LLM    = "] Asking the LLM for the other relations"
TO_CHECK      = re.compile(r"\] (?P<n>\d+) relation\(s\) to check$")
CHECKING      = "] Checking: "
NEW_RELATIONS = re.compile(r"\] (?P<n>\d+) new relation\(s\) added to relationships\.yaml")
# The index (offline/indexer/schema_indexer.py): the values read table after table (or kept, the data of the table
# unchanged since the last index), then the documents built and written into Chroma (the ones that changed), then
# the value index
FETCHING    = re.compile(r"\] (?:Fetching values|Values kept): (?P<table>\S+) \(")
VALUES_KEPT = re.compile(r"\] Values kept: \S+ \(")
EMBEDDED    = re.compile(r"\] Embedded (?P<n>\d+) of (?P<total>\d+) schemas")
PHASES   = (("] Built: ", "documents"), (" schemas to Chroma.", "values"))
# A question of the eval asked to the agent (offline/eval/run.py)
QUESTION = re.compile(r"^\[(?P<n>\d+)/(?P<total>\d+)\] ")
# A table described again (semantics --fresh): the descriptions the LLM changed, the ones corrected by hand kept
REWRITTEN = re.compile(r"\] \S+: (?P<rewritten>\d+) description\(s\) rewritten, to read again(?:; (?P<kept>\d+) corrected by hand kept)?$")
# The columns the LLM described without being able to tell their meaning or unit, to state by hand
UNCLEAR   = re.compile(r"\] (?P<unclear>\d+) column description\(s\) unclear, to state by hand$")
# Lines of offline/query_cancel.py on a stop
CANCELLED     = "the running query is cancelled on the cluster"
NOT_CANCELLED = "the running query could not be cancelled"

# The steps, as the message of a job already running names them
STEP_NAMES = {"profile": "profilage", "semantics": "description", "relations": "relations",
              "metrics": "métriques", "index": "index", "eval": "évaluation"}


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
    to_check: int | None = None                               # ... out of the relations to measure
    added: int | None = None                                  # relations: the new ones the search added
    phase: str | None = None                                  # index: "documents", then "values" (PHASES);
                                                              # relations: "llm", then "check"
    progress: list[int] | None = None                         # eval: [question, questions]
    rewritten: int = 0                                        # semantics --fresh: descriptions the LLM changed
    kept: int = 0                                             # ... and the ones corrected by hand it kept
    unclear: int = 0                                          # semantics: columns described, their meaning unclear
    detected: dict = field(default_factory=lambda: {"formats": 0, "empty": 0, "personal": 0})   # profile: summed
    values_kept: int = 0                                      # index: tables whose values were kept (data unchanged)
    embedded: list[int] | None = None                         # index: [documents embedded, documents], when not all
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
            self.values_kept += bool(VALUES_KEPT.search(message))
            if self.current and self.current not in self.failed:
                self.done.append(self.current)
            self.current, self.current_since = started["table"], time.time()
        elif phase:
            # The values are all read: the last table is done
            if self.current and self.current not in self.failed:
                self.done.append(self.current)
            self.current, self.current_since, self.phase = None, None, phase
        elif embedded := EMBEDDED.search(message):
            self.embedded = [int(embedded["n"]), int(embedded["total"])]
        elif question := QUESTION.match(message):
            self.progress = [int(question["n"]), int(question["total"])]
        elif described := REWRITTEN.search(message):
            self.rewritten += int(described["rewritten"])
            self.kept      += int(described["kept"] or 0)
        elif doubted := UNCLEAR.search(message):
            self.unclear += int(doubted["unclear"])
        elif failed := FAILED.search(message):
            self.failed.append(failed["table"])
        elif CHECKING in message:
            self.checked += 1
        elif detected := DETECTED.search(message):
            for kind in self.detected:
                self.detected[kind] += int(detected[kind])
        elif ASKING_LLM in message:
            self.phase = "llm"
        elif to_check := TO_CHECK.search(message):
            self.phase, self.to_check = "check", int(to_check["n"])
        elif added := NEW_RELATIONS.search(message):
            self.added = int(added["n"])
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
            "rewritten": self.rewritten, "kept": self.kept, "unclear": self.unclear,
            "detected": self.detected, "to_check": self.to_check, "added": self.added,
            "values_kept": self.values_kept, "embedded": self.embedded,
            "lines": self.lines[start:], "next_line": self.first_line + len(self.lines),
        }


def profile_command(db: str, tables: list[str] | None, missing: bool, complete: bool = False) -> list[str]:
    """
    The CLI command of `make profile`: the tables asked for, or the ones missing from the profile; with
    complete, only what an older profile of those tables (all of them without any) lacks.
    """
    command = [sys.executable, "-m", "offline.cli", "profile", "--db", db]
    if tables:
        command += ["--tables", ",".join(tables)]
    if complete:
        command.append("--complete")
    elif missing and not tables:
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


def semantics_command(db: str, tables: list[str] | None, fresh: bool, replace_edited: bool = False) -> list[str]:
    """
    The CLI command of `make semantics`: the missing descriptions, or `tables` described again (fresh),
    the descriptions corrected by hand kept unless replace_edited.
    """
    command = [sys.executable, "-m", "offline.cli", "semantics", "--db", db]
    if tables:
        command += ["--tables", ",".join(tables)]
    if fresh:
        command.append("--fresh")
        if replace_edited:
            command.append("--replace-edited")
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
                running = STEP_NAMES.get(self.job.step, self.job.step)
                raise RuntimeError(f"Un job tourne déjà : {running} sur {self.job.db}")
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
