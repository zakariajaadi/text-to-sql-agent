# offline/query_cancel.py

"""
Stopping the CLI cancels its running query on the cluster: offline/studio stops a job with SIGTERM,
Ctrl+C sends SIGINT. Killing the process alone only closes its socket: HiveServer2 and Impala keep
running the query, and a few of them left behind are enough to freeze a small Hive.

The queries of Hive and Impala are run asynchronously and polled here (pyhive runs them synchronously,
without the handle a cancel needs). A stop signal raises KeyboardInterrupt while the thread waits
between two polls, and only sets a flag during a call to the server, read once the call returns: the
cancel then goes over the same connection, never in the middle of another call.
"""

import signal
import threading
import time

from loguru import logger
from sqlalchemy import event

POLL_SECONDS = 0.5

_stop  = threading.Event()   # a stop signal was received
_local = threading.local()   # in_rpc: the thread is in a call to the server


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
# Install
# ------------------------------------------------------------

def install(engines) -> None:
    """Catch SIGTERM and SIGINT, and run the queries of the Hive and Impala engines through _execute."""
    signal.signal(signal.SIGTERM, _on_stop_signal)
    signal.signal(signal.SIGINT, _on_stop_signal)
    for engine in engines:
        if engine.dialect.name in DRIVERS:
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
    """do_execute of SQLAlchemy: start the query, poll it, cancel it on a stop. True: executed here."""
    driver = DRIVERS[context.dialect.name]
    if _stop.is_set():
        raise KeyboardInterrupt
    _rpc(driver.start, cursor, statement, parameters)
    try:
        while _rpc(driver.running, cursor):
            if _stop.is_set():
                raise KeyboardInterrupt
            time.sleep(POLL_SECONDS)
        _rpc(driver.finish, cursor)
    except KeyboardInterrupt:
        _cancel(driver, cursor)
        raise
    return True


def _cancel(driver, cursor) -> None:
    try:
        _rpc(driver.cancel, cursor)
        logger.warning("Stopped: the running query is cancelled on the cluster")
    except Exception as e:
        logger.warning(f"Stopped, but the running query could not be cancelled, it may keep running on the cluster — {e}")
