# offline/doctor.py

"""
Health check of a machine running the app in Docker (docker-compose.ubi.yaml, `make -f Makefile.ubi doctor`),
in two layers:
  - TCP: a connection opens to Impala (IMPALA_IP:IMPALA_PORT) and the Kerberos KDC (KDC_IP:88).
  - SELECT 1: a real Kerberos query on every coordinator of IMPALA_COORDINATORS ("host=ip,host=ip";
    without it, the single IMPALA_HOST/IMPALA_IP). A hung impalad accepts the TCP connection and then
    never answers the handshake, so TCP alone says "OK" while the app times out; querying every
    coordinator also names the one to switch IMPALA_HOST to. The listed names are resolved in-process,
    so coordinators beyond the .env one need no extra_hosts entry.

Exit code 1 when Impala or the KDC is unreachable over TCP, or the .env coordinator fails SELECT 1.
Another coordinator failing is reported but doesn't fail the check: the app doesn't use it.
"""

import os
import socket
import sys
import time

TCP_TIMEOUT_S = 5
QUERY_TIMEOUT_S = 10


def probe(host: str, port: int) -> str:
    """'OK' when a TCP connection to host:port opens within TCP_TIMEOUT_S, 'KO (<reason>)' otherwise."""
    try:
        socket.create_connection((host, port), TCP_TIMEOUT_S).close()
        return "OK"
    except OSError as e:
        return f"KO ({e})"


def coordinators() -> list[tuple[str, str]]:
    """(host, ip) pairs from IMPALA_COORDINATORS, defaulting to the single IMPALA_HOST/IMPALA_IP."""
    listed = os.getenv("IMPALA_COORDINATORS", "")
    if listed:
        pairs = []
        for entry in listed.split(","):
            host, _, ip = entry.strip().partition("=")
            if host.strip() and ip.strip():
                pairs.append((host.strip(), ip.strip()))
        return pairs
    host, ip = os.getenv("IMPALA_HOST", ""), os.getenv("IMPALA_IP", "")
    return [(host, ip)] if host and ip else []


def resolve_locally(hosts: dict[str, str]) -> None:
    """Route the listed host names to their IPs inside this process: the container's DNS doesn't
    know the cluster, and Kerberos needs the names (the service principal is impala/<host>)."""
    original = socket.getaddrinfo

    def getaddrinfo(host, *args, **kwargs):
        return original(hosts.get(host, host), *args, **kwargs)

    socket.getaddrinfo = getaddrinfo


def select_1(host: str) -> str:
    """'OK (0.8s)' when a Kerberos SELECT 1 on this coordinator answers within QUERY_TIMEOUT_S."""
    from impala.dbapi import connect  # heavy import, skipped when there is no coordinator to query

    started = time.monotonic()
    try:
        conn = connect(
            host=host,
            port=int(os.getenv("IMPALA_PORT", "21050")),
            auth_mechanism=os.getenv("IMPALA_AUTH_MECHANISM", "GSSAPI"),
            kerberos_service_name=os.getenv("IMPALA_KERBEROS_SERVICE", "impala"),
            use_ssl=os.getenv("IMPALA_USE_SSL", "false").lower() == "true",
            timeout=QUERY_TIMEOUT_S,
            retries=1,
        )
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchall()
            cursor.close()
        finally:
            conn.close()
        return f"OK ({time.monotonic() - started:.1f}s)"
    except Exception as e:  # whatever failed IS the diagnosis: report it, never crash
        reason = str(e).strip() or type(e).__name__
        return f"KO after {time.monotonic() - started:.1f}s ({reason})"


def main() -> int:
    targets = [
        ("Impala",   os.getenv("IMPALA_IP", ""), int(os.getenv("IMPALA_PORT", "21050"))),
        ("Kerberos", os.getenv("KDC_IP", ""),    88),
    ]
    tcp_failed = False
    for name, host, port in targets:
        result = probe(host, port) if host else "KO (not set in .env)"
        tcp_failed |= result != "OK"
        print(f"  {name:<9} {host}:{port}  {result}")

    nodes = coordinators()
    results = {}
    if nodes and not tcp_failed:
        resolve_locally(dict(nodes))
        configured = os.getenv("IMPALA_HOST", "")
        print("Impala answers (Kerberos SELECT 1):")
        for host, ip in nodes:
            results[host] = select_1(host)
            marker = "  <- .env" if host == configured else ""
            print(f"  {host:<40} {ip:<15} {results[host]}{marker}")

    configured_ok = results.get(os.getenv("IMPALA_HOST", ""), "OK").startswith("OK")
    if tcp_failed:
        print("The container can't reach the cluster: if Windows can (above), restart Docker Desktop (Quit, then start), else check the VPN.")
    elif not configured_ok and any(r.startswith("OK") for r in results.values()):
        print("The .env coordinator doesn't answer: switch IMPALA_HOST/IMPALA_IP to one marked OK, restart the app.")
    elif not configured_ok:
        print("No coordinator answers although TCP passes: the cluster side is down, or Docker Desktop's network is wedged (restart it).")
    else:
        print("Network OK, Impala answers.")
    return 1 if tcp_failed or not configured_ok else 0


if __name__ == "__main__":
    sys.exit(main())
