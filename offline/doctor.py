# offline/doctor.py

"""
Network check of a machine running the app in Docker (docker-compose.ubi.yaml, `make -f Makefile.ubi doctor`):
can the container reach the cluster? Impala (IMPALA_IP:IMPALA_PORT) and the Kerberos KDC (KDC_IP:88),
read from the environment (.env). Opens a TCP connection only: no query, no ticket.

Exit code 1 when one of them is unreachable.
"""

import os
import socket
import sys

TIMEOUT_S = 5


def probe(host: str, port: int) -> str:
    """'OK' when a TCP connection to host:port opens within TIMEOUT_S, 'KO (<reason>)' otherwise."""
    try:
        socket.create_connection((host, port), TIMEOUT_S).close()
        return "OK"
    except OSError as e:
        return f"KO ({e})"


def main() -> int:
    targets = [
        ("Impala",   os.getenv("IMPALA_IP", ""), int(os.getenv("IMPALA_PORT", "21050"))),
        ("Kerberos", os.getenv("KDC_IP", ""),    88),
    ]
    failed = False
    for name, host, port in targets:
        result = probe(host, port) if host else "KO (not set in .env)"
        failed |= result != "OK"
        print(f"  {name:<9} {host}:{port}  {result}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
