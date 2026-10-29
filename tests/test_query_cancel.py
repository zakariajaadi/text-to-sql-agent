"""
A stopped run cancels its running query on the cluster (lib/query_cancel.py): the CLI and the
studio jobs on a signal (KeyboardInterrupt), a turn of the app on its cancel event, set by the
Stop button (QueryCancelled). Without either, the listener steps aside.
"""

import threading
from types import SimpleNamespace

import pytest

from lib import query_cancel
from lib.query_cancel import QueryCancelled, cancel_event, discard_cancel, request_cancel


class FakeDriver:
    """A query running for `polls` polls; `on_poll` runs during a poll (a call to the server)."""

    def __init__(self, polls: int, on_poll=None):
        self.polls, self.on_poll, self.calls = polls, on_poll, []

    def start(self, cursor, statement, parameters):
        self.calls.append(("start", statement))

    def running(self, cursor):
        self.calls.append("poll")
        if self.on_poll:
            self.on_poll()
        self.polls -= 1
        return self.polls > 0

    def finish(self, cursor):
        self.calls.append("finish")

    def cancel(self, cursor):
        self.calls.append("cancel")


@pytest.fixture
def driver(monkeypatch):
    monkeypatch.setattr(query_cancel, "POLL_SECONDS", 0)
    monkeypatch.setattr(query_cancel, "_signals", True)   # as after install(): every query is polled
    query_cancel._stop.clear()
    yield lambda **kwargs: monkeypatch.setitem(query_cancel.DRIVERS, "hive", FakeDriver(**kwargs)) or query_cancel.DRIVERS["hive"]
    query_cancel._stop.clear()


def context(cancel=None) -> SimpleNamespace:
    options = {"cancel_event": cancel} if cancel is not None else {}
    return SimpleNamespace(dialect=SimpleNamespace(name="hive"), execution_options=options)


def test_a_query_runs_until_done(driver):
    fake = driver(polls=3)
    assert query_cancel._execute(object(), "SELECT 1", {}, context()) is True
    assert fake.calls == [("start", "SELECT 1"), "poll", "poll", "poll", "finish"]


def test_a_stop_during_a_call_waits_for_it_then_cancels(driver):
    # the signal arrives during the second poll: no raise inside the call, the cancel follows it
    fake = driver(polls=5, on_poll=lambda: len(fake.calls) == 3 and query_cancel._on_stop_signal(15, None))
    with pytest.raises(KeyboardInterrupt):
        query_cancel._execute(object(), "SELECT 1", {}, context())
    assert fake.calls == [("start", "SELECT 1"), "poll", "poll", "cancel"]


def test_a_stop_between_two_calls_raises_at_once():
    query_cancel._stop.clear()
    with pytest.raises(KeyboardInterrupt):
        query_cancel._on_stop_signal(15, None)
    query_cancel._stop.clear()


def test_after_a_stop_no_query_starts(driver):
    fake = driver(polls=1)
    query_cancel._stop.set()
    with pytest.raises(KeyboardInterrupt):
        query_cancel._execute(object(), "SELECT 1", {}, context())
    assert fake.calls == []


def test_a_failed_cancel_is_said(driver, monkeypatch):
    fake = driver(polls=5, on_poll=lambda: query_cancel._stop.set())
    monkeypatch.setattr(fake, "cancel", lambda cursor: 1 / 0)
    messages = []
    monkeypatch.setattr(query_cancel.logger, "warning", messages.append)
    with pytest.raises(KeyboardInterrupt):
        query_cancel._execute(object(), "SELECT 1", {}, context())
    assert "could not be cancelled" in messages[0]


# ── The cancel event of a turn of the app (the Stop button) ───────────────────

def test_the_turns_cancel_event_cancels_the_query(driver, monkeypatch):
    monkeypatch.setattr(query_cancel, "_signals", False)   # the app installs no signals
    stop = threading.Event()
    fake = driver(polls=5, on_poll=lambda: len(fake.calls) == 3 and stop.set())
    with pytest.raises(QueryCancelled):
        query_cancel._execute(object(), "SELECT 1", {}, context(cancel=stop))
    assert fake.calls == [("start", "SELECT 1"), "poll", "poll", "cancel"]


def test_without_a_signal_install_nor_an_event_the_listener_steps_aside(driver, monkeypatch):
    monkeypatch.setattr(query_cancel, "_signals", False)
    fake = driver(polls=1)
    # None: SQLAlchemy runs the query itself, the driver is never called
    assert query_cancel._execute(object(), "SELECT 1", {}, context()) is None
    assert fake.calls == []


def test_the_registry_of_the_turns():
    try:
        turn_event = cancel_event("turn-1")
        assert cancel_event("turn-1") is turn_event and not turn_event.is_set()
        request_cancel("turn-1")
        assert turn_event.is_set()
        # an unknown key sets nothing and does not fail; the thread keeps the event it holds
        request_cancel("turn-2")
        discard_cancel("turn-1")
        assert cancel_event("turn-1") is not turn_event and turn_event.is_set()
    finally:
        discard_cancel("turn-1")
        discard_cancel("turn-2")
