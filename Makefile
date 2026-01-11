.PHONY: dev lint type test watch logs help

help:
	@echo "Available targets:"
	@echo "  make dev     - Install dependencies"
	@echo "  make watch   - Run development server with auto-reload"
	@echo "  make logs    - Tail and format application logs"
	@echo "  make test    - Run test suite"
	@echo "  make lint    - Run linter (ruff)"
	@echo "  make type    - Run type checker (mypy)"

dev:
	python -m pip install -U pip
	pip install -r requirements.txt || true

watch:
	@echo "Starting development server with auto-reload..."
	@echo "Access at http://localhost:5050"
	@echo "Press Ctrl+C to stop"
	PYTHONPATH=.. venv/bin/uvicorn voice_agent_v4.app:app --host 0.0.0.0 --port 5050 --reload --log-level info

logs:
	@mkdir -p logs
	@echo "Tailing logs with trace viewer (Ctrl+C to exit)..."
	@echo "Filters: --call SID --event EVENT_NAME --since 5m"
	venv/bin/python scripts/tail_traces.py

lint:
	venv/bin/ruff check .

type:
	venv/bin/mypy app.py actor.py config.py events.py transport_twilio.py core/ vad_modular/ providers/ observability/ mcp_tools/ tests/

test:
	PYTHONPATH=..:. venv/bin/pytest tests/ --ignore=tests/scripts/ -q
