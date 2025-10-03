# TradeML Makefile
# Common commands for development and deployment

.PHONY: help setup docker-up docker-down docker-logs test validate clean data fetch-sample edge-up curator-up dev-s3

help:
	@echo "TradeML - Autonomous Trading Agent"
	@echo ""
	@echo "Available commands:"
	@echo "  make setup          - Set up Python environment and dependencies"
	@echo "  make docker-up      - Start Docker services (PostgreSQL, MinIO, etc.)"
	@echo "  make docker-down    - Stop Docker services"
	@echo "  make docker-logs    - View Docker service logs"
	@echo "  make edge-up        - Start edge collector (with dev MinIO)"
	@echo "  make curator-up     - Start curator (with dev MinIO)"
	@echo "  make dev-s3         - Start only MinIO (dev S3)"
	@echo "  make validate       - Run setup validation checks"
	@echo "  make test           - Run unit tests"
	@echo "  make test-int       - Run integration tests"
	@echo "  make lint           - Run linters (black, ruff, mypy)"
	@echo "  make format         - Format code with black"
	@echo "  make fetch-sample   - Fetch sample data (requires API keys)"
	@echo "  make clean          - Clean temporary files and caches"
	@echo ""

# Setup Python environment
setup:
	@echo "Setting up Python environment..."
	python -m venv venv
	@echo "Activate with: source venv/bin/activate (or venv\Scripts\activate on Windows)"
	@echo "Then run: make install"

install:
	@echo "Installing dependencies..."
	pip install --upgrade pip
	pip install -r requirements.txt
	@echo "✓ Dependencies installed"

# Docker operations
docker-up:
	@echo "Starting Docker services..."
	cd infra && docker-compose up -d
	@echo "✓ Services started"
	@echo "PostgreSQL:  localhost:5432"
	@echo "MinIO:       http://localhost:9001 (minioadmin/minioadmin)"
	@echo "MLflow:      http://localhost:5000"
	@echo "Redis:       localhost:6379"

docker-down:
	@echo "Stopping Docker services..."
	cd infra && docker-compose down
	@echo "✓ Services stopped"

docker-logs:
	cd infra && docker-compose logs -f

docker-ps:
	cd infra && docker-compose ps

# Edge/Curator services via compose profiles
edge-up:
	@echo "Starting edge collector with dev MinIO..."
	cd infra && COMPOSE_PROFILES=dev-s3,edge docker-compose up -d minio minio-init edge-collector
	@echo "✓ Edge collector running (profile: edge + dev-s3)"
	@echo "MinIO Console: http://localhost:9001 (minioadmin/minioadmin)"

curator-up:
	@echo "Starting curator with dev MinIO..."
	cd infra && COMPOSE_PROFILES=dev-s3,curator docker-compose up -d minio minio-init curator
	@echo "✓ Curator running (profile: curator + dev-s3)"

dev-s3:
	@echo "Starting dev MinIO only..."
	cd infra && COMPOSE_PROFILES=dev-s3 docker-compose up -d minio minio-init
	@echo "✓ MinIO started at http://localhost:9001 (minioadmin/minioadmin)"

# Validation
validate:
	@echo "Running setup validation..."
	python setup.py

# Testing
test:
	@echo "Running unit tests..."
	pytest tests/unit -v --cov=data_layer --cov=models --cov=validation

test-int:
	@echo "Running integration tests (requires Docker)..."
	pytest tests/integration -v

test-accept:
	@echo "Running acceptance tests..."
	pytest tests/acceptance -v

test-all:
	@echo "Running all tests..."
	pytest tests/ -v --cov=. --cov-report=html

# Smoke: one-cycle collect -> curate -> audit summary (local storage)
smoke:
	@echo "Running one-cycle smoke (local storage)..."
	STORAGE_BACKEND=local python scripts/smoke.py
	@echo "✓ Smoke complete"

# Code quality
lint:
	@echo "Running linters..."
	black --check .
	ruff check .
	mypy data_layer/ models/ validation/ portfolio/ execution/

format:
	@echo "Formatting code..."
	black .
	@echo "✓ Code formatted"

# Data operations
fetch-sample:
	@echo "Fetching sample data (last 1 year, daily bars)..."
	python -m data_layer.connectors.alpaca_connector \
		--symbols AAPL MSFT GOOGL AMZN NVDA META TSLA \
		--start-date 2023-01-01 \
		--end-date 2024-01-01 \
		--timeframe 1Day \
		--output data_layer/raw/equities_bars
	@echo "✓ Sample data fetched"

# Cleaning
clean:
	@echo "Cleaning temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@echo "✓ Cleaned"

clean-data:
	@echo "WARNING: This will delete all downloaded data!"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf data_layer/raw/*; \
		rm -rf data_layer/curated/*; \
		echo "✓ Data cleaned"; \
	fi

# Database operations
db-init:
	@echo "Initializing database schema..."
	docker exec -i trademl_postgres psql -U trademl -d trademl < infra/init-db/01-init-schema.sql
	@echo "✓ Database initialized"

db-reset:
	@echo "WARNING: This will reset the database!"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		cd infra && docker-compose down -v; \
		docker-compose up -d postgres; \
		sleep 5; \
		make db-init; \
		echo "✓ Database reset"; \
	fi

db-shell:
	docker exec -it trademl_postgres psql -U trademl -d trademl

# MLflow
mlflow-ui:
	@echo "Opening MLflow UI..."
	@echo "Navigate to: http://localhost:5000"

# Environment
env-setup:
	@if [ ! -f .env ]; then \
		cp .env.template .env; \
		echo "✓ Created .env from template"; \
		echo "Edit .env and add your API keys"; \
	else \
		echo ".env already exists"; \
	fi

# Development
dev-check: lint test
	@echo "✓ All checks passed"

# Quick start (run this first)
quickstart: env-setup docker-up install validate
	@echo ""
	@echo "✓ TradeML setup complete!"
	@echo ""
	@echo "Next steps:"
	@echo "1. Edit .env with your API keys"
	@echo "2. Run: make validate"
	@echo "3. Run: make fetch-sample"
	@echo "4. Review: README.md (Quick Start) and QUICK_REFERENCE.md"
