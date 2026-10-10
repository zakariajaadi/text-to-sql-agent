# offline/__init__.py

# Offline tools run by the developer, never by the app: semantic layer generators, schema indexer,
# eval and Hive loading. They import the app code from src/ (PYTHONPATH=src); src/ never imports them.
