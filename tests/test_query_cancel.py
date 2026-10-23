"""A stopped CLI cancels its running query on the cluster (offline/query_cancel.py)."""

from types import SimpleNamespace

import pytest

from offline import query_cancel


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
    query_cancel._stop.clear()
    yield lambda **kwargs: monkeypatch.setitem(query_cancel.DRIVERS, "hive", FakeDriver(**kwargs)) or query_cancel.DRIVERS["hive"]
    query_cancel._stop.clear()


CONTEXT = SimpleNamespace(dialect=SimpleNamespace(name="hive"))


def test_a_query_runs_until_done(driver):
    fake = driver(polls=3)
    assert query_cancel._execute(object(), "SELECT 1", {}, CONTEXT) is True
    assert fake.calls == [("start", "SELECT 1"), "poll", "poll", "poll", "finish"]


def test_a_stop_during_a_call_waits_for_it_then_cancels(driver):
    # the signal arrives during the second poll: no raise inside the call, the cancel follows it
    fake = driver(polls=5, on_poll=lambda: len(fake.calls) == 3 and query_cancel._on_stop_signal(15, None))
    with pytest.raises(KeyboardInterrupt):
        query_cancel._execute(object(), "SELECT 1", {}, CONTEXT)
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
        query_cancel._execute(object(), "SELECT 1", {}, CONTEXT)
    assert fake.calls == []


def test_a_failed_cancel_is_said(driver, monkeypatch):
    fake = driver(polls=5, on_poll=lambda: query_cancel._stop.set())
    monkeypatch.setattr(fake, "cancel", lambda cursor: 1 / 0)
    messages = []
    monkeypatch.setattr(query_cancel.logger, "warning", messages.append)
    with pytest.raises(KeyboardInterrupt):
        query_cancel._execute(object(), "SELECT 1", {}, CONTEXT)
    assert "could not be cancelled" in messages[0]
