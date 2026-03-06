.PHONY: install run test-data verify clean help

PYTHON = python3
VENV = venv
BIN = $(VENV)/bin
PIP = $(BIN)/pip
UVICORN = $(BIN)/uvicorn
PROJECT = weedsVsCrops
EPOCHS = 10
IMGSZ = 640
BATCH = 16

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
	. $(VENV)/bin/activate && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --reload-exclude "datasets/*" --reload-exclude "models/*" --reload-exclude "runs/*" --reload-exclude "*.yaml"

test-data: install
	$(BIN)/python setup_test_data.py

convert-coco-txt: install
	$(BIN)/python convert_coco_txt.py

verify: test-data
	@echo "Ensure the server is running on http://localhost:8000 before running this."
	@echo "1. Testing dataset info..."
	curl -s "http://127.0.0.1:8000/api/v1/training/dataset/info?project=$(PROJECT)" | jq .
	@echo "\n2. Starting training..."
	curl -s -X POST "http://127.0.0.1:8000/api/v1/training/start" \
	     -H "Content-Type: application/json" \
	     -d "{\"project_name\": \"$(PROJECT)\", \"epochs\": 5, \"imgsz\": 640, \"batch\": 16}" | jq .
	@echo "\n3. Waiting for training to complete (polling status)..."
	@status="training"; \
	while [ "$$status" = "training" ] || [ "$$status" = "idle" ]; do \
		response=$$(curl -s "http://127.0.0.1:8000/api/v1/training/status?project=$(PROJECT)"); \
		status=$$(echo $$response | jq -r .status); \
		progress=$$(echo $$response | jq -r .progress); \
		epoch=$$(echo $$response | jq -r .current_epoch); \
		total=$$(echo $$response | jq -r .total_epochs); \
		message=$$(echo $$response | jq -r .message); \
		if [ "$$status" = "training" ]; then \
			echo "Status: $$status | Progress: $$progress% | Epoch: $$epoch/$$total | $$message"; \
		else \
			echo "Status: $$status | $$message"; \
		fi; \
		if [ "$$status" = "completed" ]; then break; fi; \
		if [ "$$status" = "failed" ]; then echo "Training FAILED!"; exit 1; fi; \
		sleep 10; \
	done
	@echo "Training completed! weights verified via status."
	@echo "\n4. Testing inference with trained model..."
	@IMAGE=$$(ls datasets/$(PROJECT)/val/images/*.jpg | head -n 1); \
	curl -s -X POST "http://127.0.0.1:8000/api/v1/detect/image?model_path=models/$(PROJECT)/weights/best.pt&confidence_threshold=0.25" \
	     -F "file=@$$IMAGE" | jq .

train: test-data
	@echo "Starting local training for project: $(PROJECT)..."
	PYTHONPATH=. $(BIN)/python train_local.py --project $(PROJECT) --epochs $(EPOCHS) --imgsz $(IMGSZ) --batch $(BATCH) $(EXTRA_ARGS)

ui: install
	. $(VENV)/bin/activate && streamlit run streamlit_app.py



clean:
	rm -rf $(VENV)
	rm -rf datasets/
	rm -rf models/*/
	rm -rf runs/
	find . -type d -name "__pycache__" -exec rm -rf {} +
