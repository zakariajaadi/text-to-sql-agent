"""
Chroma needs sqlite3 >= 3.35, newer than the system one of RHEL 8 / UBI 8 images (dockerfile.ubi).
There, pysqlite3-binary (the `sqlite` extra) brings its own build of SQLite, swapped in for the
standard sqlite3 module. Imported first by lib/__init__.py, so before any lib module imports
sqlite3 (value_index) or chromadb (vector_store). Elsewhere pysqlite3 is not installed: no-op.
"""

import sys

try:
    import pysqlite3
except ImportError:
    pass
else:
    sys.modules["sqlite3"] = pysqlite3
