# Usage: make <target> [DB=Chinook]
# Typical first run: hive-up → hive-load → profile → semantics → relations → metrics → (review) → index → app

# ==============================================================================
# Variables
# ==============================================================================
DB        ?= Chinook
HIVE_DB    = $(shell echo $(DB) | tr A-Z a-z)
HIVE_JDBC  = jdbc:hive2://localhost:10000
RUN        = PYTHONPATH=src uv run python

.DEFAULT_GOAL := help
.PHONY: help \
        hive-up hive-check hive-ping hive-load hive-partitioned \
        profile semantics relations metrics index schema studio \
        app app-docker app-logs app-stop \
        eval eval-dashboard eval-show test lint

##@ Help

help:  ## Show available targets, by section
	@awk 'BEGIN {FS = ":.*## "} \
		/^##@/ {printf "\n\033[1m%s\033[0m\n", substr($$0, 5)} \
		/^[a-z-]+:.*## / {printf "  \033[36m%-17s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

##@ Hive (local)

hive-up:  ## Start the local Hive container
	docker compose up -d hive

hive-check:  ## Check HiveServer2 is up (beeline, inside the container)
	docker exec hive beeline -u $(HIVE_JDBC) -e "SHOW DATABASES;"

hive-ping:  ## Check the Python side: SQLAlchemy + PyHive connection
	uv run python -c "from langchain_community.utilities import SQLDatabase; \
	db = SQLDatabase.from_uri('hive://hive@localhost:10000/default'); \
	print('dialect:', db.dialect); print(db.run('SELECT 1 + 1'))"

hive-load:  ## Load a SQLite db into Hive (DB=Chinook)
	uv run python offline/hive/export_to_hive.py $(DB)
	docker cp data/hive_load/. hive:/tmp/hive_load
	docker exec hive beeline -u $(HIVE_JDBC) -f /tmp/hive_load/$(HIVE_DB)/load.hql

hive-partitioned:  ## Create the read-mode test tables (Full/Last, Delta/All, CDC/Live)
	docker cp offline/hive/partitioned_tables.hql hive:/tmp/partitioned_tables.hql
	docker exec hive beeline -u $(HIVE_JDBC) -f /tmp/partitioned_tables.hql

##@ Semantic layer: profile → semantics → relations → metrics → (review) → index

profile:  ## Measure the data → semantic_layer/<DB>/profile.yaml (no LLM; ARGS="--tables a,b" or "--missing": only those, merged)
	$(RUN) -m offline.cli profile --db $(DB) $(ARGS)

semantics:  ## Generate table/column descriptions and categorical columns (ARGS="--tables a,b" and/or "--fresh")
	$(RUN) -m offline.cli semantics --db $(DB) $(ARGS)

relations:  ## Generate FK relations, each checked on the data (ARGS="--check": only check them again, no LLM)
	$(RUN) -m offline.cli fk-relations --db $(DB) $(ARGS)

metrics:  ## Propose business metrics (review semantic_layer/<DB>/metrics.yaml)
	$(RUN) -m offline.cli metrics --db $(DB)

index:  ## Index the tables into Chroma and the categorical values into data/values/<DB>.sqlite (no LLM)
	$(RUN) -m offline.cli index --db $(DB)

schema:  ## Show the indexed document of a table (TABLE=invoice [DB=Chinook])
	$(if $(TABLE),,$(error TABLE is required, e.g. make schema TABLE=invoice))
	@$(RUN) -c "from resources import vector_store; \
	docs = vector_store.get_store().get(ids=['$(DB)_$(TABLE)'])['documents']; \
	print(docs[0] if docs else 'Not found: $(DB)_$(TABLE)')"

studio:  ## Run the studio of the semantic layer (run the steps, review their proposals) on http://localhost:8100
	$(RUN) -m uvicorn offline.studio.app:app --port 8100 --reload --reload-dir offline/studio

##@ App

app:  ## Run the Chainlit app locally, on http://localhost:8000
	uv run chainlit run app.py

app-docker:  ## Build and run the app in Docker (with Redis and Hive), on http://localhost:8000
	docker compose up -d --build app

app-logs:  ## Follow the app container logs
	docker compose logs -f app

app-stop:  ## Stop the app container (Hive and Redis keep running)
	docker compose stop app

##@ Quality

eval:  ## Run the eval on DB, results in offline/eval/results/<db>.json (options: ARGS="--limit 5")
	$(RUN) -m offline.eval.run --db $(DB) $(ARGS)

eval-dashboard:  ## Build the eval dashboard, offline/eval/results/dashboard.html (options: ARGS="--db Chinook Northwind")
	$(RUN) -m offline.eval.dashboard $(ARGS)

eval-show:  ## Print the eval misses of DB as text: agent SQL, accepted queries, judge (ARGS="--only Brazil": those questions)
	$(RUN) -m offline.eval.show --db $(DB) $(ARGS)

test:  ## Run the unit tests
	uv run pytest

lint:  ## Lint with ruff
	uv run ruff check src offline
