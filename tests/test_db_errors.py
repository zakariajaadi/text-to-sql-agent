"""An unreachable database told from a query error (src/lib/db_errors.py)."""

import socket

import pytest
from sqlalchemy.exc import DBAPIError, OperationalError, ProgrammingError
from thrift.transport.TTransport import TTransportException

from lib.db_errors import is_unreachable


class GSSError(Exception):
    """Same name as the kerberos package's: matched by name, the package is optional."""


class HiveServer2Error(Exception):
    """An error the server returned about the query, like impyla's."""


def wrapped(error: Exception) -> DBAPIError:
    """The error as SQLAlchemy raises it, the driver's one in .orig."""
    return OperationalError("SELECT 1", {}, error)


@pytest.mark.parametrize("error", [
    TTransportException(message="Could not connect to any of [('10.0.0.1', 21050)]"),
    ConnectionRefusedError(111, "Connection refused"),
    socket.timeout("timed out"),
    TimeoutError(),
    GSSError("Cannot contact any KDC for realm"),
    wrapped(TTransportException(message="TSocket read 0 bytes")),
    wrapped(OSError("Network is unreachable")),
    wrapped(Exception("Invalid SessionHandle: SessionHandle [91eed306-4b6d-4718-b2a0-c64c15cdc506]")),
])
def test_unreachable_database(error):
    assert is_unreachable(error)


def test_an_error_caused_by_a_network_error_is_unreachable():
    try:
        try:
            raise ConnectionResetError("Connection reset by peer")
        except ConnectionResetError as cause:
            raise RuntimeError("query failed") from cause
    except RuntimeError as error:
        assert is_unreachable(error)


@pytest.mark.parametrize("error", [
    HiveServer2Error("AnalysisException: Could not resolve column/field reference: 'x'"),
    wrapped(HiveServer2Error("ParseException: Syntax error in line 1")),
    ProgrammingError("SELECT x", {}, Exception("unknown column")),
    HiveServer2Error("Query 1a:2b expired due to execution time limit of 120s000ms"),
    ValueError("bad value"),
])
def test_query_errors_are_not_unreachable(error):
    assert not is_unreachable(error)
