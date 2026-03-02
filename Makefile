.PHONY: install run test-data verify clean help

PYTHON = python3
VENV = venv
BIN = $(VENV)/bin
PIP = $(BIN)/pip
UVICORN = $(BIN)/uvicorn

help:
	@echo "Usage:"
	@echo "  make install    - Setup virtual environment and install dependencies"
	@echo "  make run        - Run the FastAPI application"
	@echo "  make test-data  - Generate test datasets"
	@echo "  make verify     - Run automated verification tests via curl"
	@echo "  make clean      - Remove venv, datasets, and generated runs"

install: $(VENV)/bin/activate

$(VENV)/bin/activate: requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	touch $(VENV)/bin/activate

run: install
	$(UVICORN) app.main:app --host 0.0.0.0 --port 8000 --reload

test-data: install
	$(BIN)/python setup_test_data.py

verify:
	@echo "Ensure the server is running on http://localhost:8000 before running this."
	@echo "Testing valid_project info..."
	curl -s "http://127.0.0.1:8000/api/v1/training/dataset/info?project=valid_project" | jq .
	@echo "\nTesting missing_label_project info (Expect 400)..."
	curl -s "http://127.0.0.1:8000/api/v1/training/dataset/info?project=missing_label_project" | jq .
	@echo "\nTesting training start..."
	curl -s -X POST "http://127.0.0.1:8000/api/v1/training/start" \
	     -H "Content-Type: application/json" \
	     -d '{"project_name": "valid_project", "epochs": 1, "imgsz": 640, "batch": 16}' | jq .

clean:
	rm -rf $(VENV)
	rm -rf datasets/valid_project datasets/missing_label_project datasets/invalid_polygon_project datasets/out_of_bounds_project datasets/empty_label_project datasets/non_contiguous_project
	rm -rf runs/
	find . -type d -name "__pycache__" -exec rm -rf {} +
