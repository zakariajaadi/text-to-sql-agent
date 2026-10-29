# src/lib/query_cancel.py

"""
Cancelling a running Hive or Impala query on the cluster. Killing a process or closing a socket
only drops the client side: HiveServer2 and Impala keep running the query, and a few of them left
behind are enough to freeze a small Hive.

The queries are run asynchronously and polled here (pyhive runs them synchronously, without the
handle a cancel needs). Two things can stop one, both read between two polls, so the cancel goes
over the same connection, never in the middle of another call to the server:

  - a stop signal of the process (install: the CLI and the studio jobs run every query this way):
    SIGTERM or Ctrl+C raises KeyboardInterrupt and cancels the running query;
  - the cancel event of one turn of the app (cancel_event(key), set by request_cancel(key) when
    the user clicks Stop, see app.py): run_capped_query passes it as the `cancel_event` execution
    option, and the query ends on QueryCancelled.

Without a signal install and without a cancel event (the eval, the generators), the listener steps
aside: SQLAlchemy runs the query as if it were not there.
"""

import signal
import threading
import time

from loguru import logger
from sqlalchemy import event

POLL_SECONDS = 0.5

_stop    = threading.Event()   # a stop signal was received (CLI, studio jobs)
_local   = threading.local()   # in_rpc: the thread is in a call to the server
_signals = False               # install() ran: every query is polled, so Ctrl+C can cancel it

# The cancel event of each turn of the app, created by cancel_event, set by request_cancel
_events:      dict[str, threading.Event] = {}
_events_lock = threading.Lock()


class QueryCancelled(Exception):
    """The turn was stopped by the user: its running query was cancelled on the cluster."""


# ------------------------------------------------------------
# Drivers: start a query, tell whether it runs, end it, cancel it
# ------------------------------------------------------------

class _Hive:
    """pyhive: ExecuteStatement with runAsync, then GetOperationStatus until the query is done."""
    RUNNING = {0, 1, 7}   # TOperationState: INITIALIZED, RUNNING, PENDING
    ERROR   = 5

    @staticmethod
    def start(cursor, statement, parameters):
        cursor.execute(statement, parameters, async_=True)

    @classmethod
    def running(cls, cursor) -> bool:
        from pyhive.exc import OperationalError
        status = cursor.poll(get_progress_update=False)
        if status.operationState == cls.ERROR:
            raise OperationalError(status.errorMessage or "Operation is in ERROR_STATE")
        return status.operationState in cls.RUNNING

    @staticmethod
    def finish(cursor):
        pass

    @staticmethod
    def cancel(cursor):
        cursor.cancel()


class _Impala:
    """impyla: execute_async, is_executing, then its own last status call, which raises the error of a failed query."""

    @staticmethod
    def start(cursor, statement, parameters):
        cursor.execute_async(statement, parameters)

    @staticmethod
    def running(cursor) -> bool:
        return cursor.is_executing()

    @staticmethod
    def finish(cursor):
        cursor._wait_to_finish()   # private: what execute() does after execute_async, one call once done

    @staticmethod
    def cancel(cursor):
        cursor.cancel_operation()


DRIVERS = {"hive": _Hive, "impala": _Impala, "impala4": _Impala}


# ------------------------------------------------------------
# The cancel event of a turn of the app
# ------------------------------------------------------------

def cancel_event(key: str) -> threading.Event:
    """The cancel event of a turn: passed to run_capped_query, set by request_cancel."""
    with _events_lock:
        return _events.setdefault(key, threading.Event())


def request_cancel(key: str) -> None:
    """Set a turn's cancel event (the user clicked Stop): its own thread cancels the query it polls."""
    with _events_lock:
        turn_event = _events.get(key)
    if turn_event:
        turn_event.set()


def discard_cancel(key: str) -> None:
    """Forget a turn's event once the turn is over; a thread still holding it keeps seeing its state."""
    with _events_lock:
        _events.pop(key, None)


# ------------------------------------------------------------
# Install
# ------------------------------------------------------------

def install(engines) -> None:
    """Catch SIGTERM and SIGINT and poll every query of the engines, so a stop cancels the running one (CLI, studio jobs)."""
    global _signals
    _signals = True
    signal.signal(signal.SIGTERM, _on_stop_signal)
    signal.signal(signal.SIGINT, _on_stop_signal)
    for engine in engines:
        install_engine(engine)


def install_engine(engine) -> None:
    """Run the Hive / Impala queries of an engine through _execute, once per engine (resources.py, install)."""
    if engine.dialect.name not in DRIVERS or getattr(engine, "_query_cancel_installed", False):
        return
    engine._query_cancel_installed = True
    event.listen(engine, "do_execute", _execute)
    event.listen(engine, "do_execute_no_params",
                 lambda cursor, statement, context: _execute(cursor, statement, None, context))


def _on_stop_signal(signum, frame):
    """Runs in the main thread: stop at once, unless it is in a call to the server (see the module)."""
    _stop.set()
    if not getattr(_local, "in_rpc", False):
        raise KeyboardInterrupt


def _rpc(call, *args):
    _local.in_rpc = True
    try:
        return call(*args)
    finally:
        _local.in_rpc = False


def _execute(cursor, statement, parameters, context):
    """
    do_execute of SQLAlchemy: start the query, poll it, cancel it on a stop signal or on the
    turn's cancel event. None when neither can happen (no signal install, no event): SQLAlchemy
    then runs the query itself. True: executed here.
    """
    cancelled = (getattr(context, "execution_options", None) or {}).get("cancel_event")
    if not _signals and cancelled is None:
        return None
    _raise_if_stopping(cancelled)
    driver = DRIVERS[context.dialect.name]
    _rpc(driver.start, cursor, statement, parameters)
    try:
        while _rpc(driver.running, cursor):
            _raise_if_stopping(cancelled)
            time.sleep(POLL_SECONDS)
        _rpc(driver.finish, cursor)
    except (KeyboardInterrupt, QueryCancelled):
        _cancel(driver, cursor)
        raise
    return True


def _raise_if_stopping(cancelled: threading.Event | None) -> None:
    """KeyboardInterrupt on a process stop (CLI), QueryCancelled on the turn's cancel event (app)."""
    if _stop.is_set():
        raise KeyboardInterrupt
    if cancelled is not None and cancelled.is_set():
        raise QueryCancelled("the turn was stopped: its running query is cancelled on the cluster")


def _cancel(driver, cursor) -> None:
    try:
        _rpc(driver.cancel, cursor)
        logger.warning("Stopped: the running query is cancelled on the cluster")
    except Exception as e:
        logger.warning(f"Stopped, but the running query could not be cancelled, it may keep running on the cluster — {e}")
