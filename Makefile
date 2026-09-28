# Usage: make <target> [DB=Chinook]

# ==============================================================================
# Configuration & Variables
# ==============================================================================
DB ?= Chinook
HIVE_DB = $(shell echo $(DB) | tr A-Z a-z)
HIVE_JDBC = jdbc:hive2://localhost:10000
RUN     = PYTHONPATH=src uv run python

.PHONY: help app hive-up hive-load semantics relations index describe eval lint

help:  ## Show available targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

# ── Chainlit ─────────────────────────────────────────────────────────────
app:  ## Run the Chainlit app locally
	uv run chainlit run app.py

# ── Inspect ─────────────────────────────────────────────────────────
schema:  ## Show the indexed document of a table (TABLE=invoice [DB=Chinook])
	$(if $(TABLE),,$(error TABLE is required, e.g. make schema TABLE=invoice))
	@$(RUN) -c "from config import cfg, embedding_model; \
	from schema_retrieval.vector_store import VectorStoreManager; \
	store = VectorStoreManager(embedding_model, cfg.retrieval.persist_directory).get_store(); \
	docs = store.get(ids=['$(DB)_$(TABLE)'])['documents']; \
	print(docs[0] if docs else 'Not found: $(DB)_$(TABLE)')"

# ── Schema pipeline: semantics → relations → (review) → index ───────
semantics:  ## Generate table/column descriptions (review config/semantics/ after)
	$(RUN) -m schema_retrieval.schema_indexer semantics --db $(DB)

relations:  ## Generate FK relations (review config/fk_relations.yaml after)
	$(RUN) -m schema_retrieval.schema_indexer fk-relations --db $(DB)

index:  ## Index schemas into Chroma (no LLM when semantics exist)
	$(RUN) -m schema_retrieval.schema_indexer --db $(DB)

describe:  ## Generate the db description for routing, if missing (review config/db_descriptions.yaml after)
	$(RUN) -m schema_retrieval.schema_indexer describe --db $(DB)

# ── Local Hive ──────────────────────────────────────────────────────
hive-up:  ## Start the local Hive container
	docker compose up -d hive

hive-load:  ## Load a SQLite db into Hive (DB=Chinook)
	uv run python scripts/export_to_hive.py $(DB)
	docker cp data/hive_load hive:/tmp/hive_load
	docker exec hive beeline -u jdbc:hive2://localhost:10000 -f /tmp/hive_load/$(HIVE_DB)/load.hql

hive-check:  ## Check HiveServer2 is up (beeline, inside the container)
	docker exec hive beeline -u $(HIVE_JDBC) -e "SHOW DATABASES;"

hive-ping:  ## Check the Python side: SQLAlchemy + PyHive connection
	uv run python -c "from langchain_community.utilities import SQLDatabase; \
	db = SQLDatabase.from_uri('hive://hive@localhost:10000/default'); \
	print('dialect:', db.dialect); print(db.run('SELECT 1 + 1'))"

# ── Quality ─────────────────────────────────────────────────────────
eval:  ## Run the eval (pass options with ARGS="--limit 5")
	uv run python eval/run.py $(ARGS)

lint:  ## Lint with ruff
	uv run ruff check src eval

