.PHONY: install dataset test eval run docker-build docker-up clean

install:
	pip install -r requirements.txt

dataset:
	cd backend && python -m app.data.generate_dataset

test:
	cd backend && python -m pytest tests/ -v

eval:
	python scripts/run_eval.py

run:
	cd backend && uvicorn app.main:app --reload --port 8000

experiments-prompt:
	cd backend && python -m app.experiments.runner prompt_version

experiments-model:
	cd backend && python -m app.experiments.runner model_tag

docker-build:
	docker compose build

docker-up:
	docker compose up

clean:
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	rm -f backend/calllens.db calllens.db
