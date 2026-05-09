.PHONY: install dev test lint format typecheck clean api worker eval

# ── Setup ──────────────────────────────────────────────────────────────────────

install:
	uv sync

install-dev:
	uv sync --dev

# ── Development ────────────────────────────────────────────────────────────────

dev:
	uv run jarvis

api:
	uv run jarvis-api

worker:
	uv run jarvis-worker

eval:
	uv run jarvis-eval

# ── Testing ────────────────────────────────────────────────────────────────────

test:
	uv run pytest tests/ -v --tb=short

test-cov:
	uv run pytest tests/ --cov=jarvis --cov-report=term-missing --cov-report=html

coverage-check:
	uv run pytest tests/ --cov=src/jarvis --cov-report=term-missing --cov-fail-under=40

test-fast:
	uv run pytest tests/ -x -q

# ── Code quality ───────────────────────────────────────────────────────────────

lint:
	uv run ruff check src/ tests/

lint-fix:
	uv run ruff check --fix src/ tests/

format:
	uv run ruff format src/ tests/

format-check:
	uv run ruff format --check src/ tests/

typecheck:
	uv run mypy src/jarvis/ --ignore-missing-imports

check: lint format-check typecheck

# ── Docker ─────────────────────────────────────────────────────────────────────

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f jarvis-api

# ── Cleanup ────────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .mypy_cache .ruff_cache .pytest_cache htmlcov .coverage
