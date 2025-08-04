.PHONY: help install test lint format train serve deploy clean setup-env quality-checks integration-tests

# Default target
help:
	@echo "Available commands:"
	@echo "  help              Show this help message"
	@echo "  install           Install dependencies"
	@echo "  setup-env         Setup development environment"
	@echo "  quality-checks    Run code quality checks (linting, formatting)"
	@echo "  test              Run unit tests"
	@echo "  integration-tests Run integration tests"
	@echo "  train             Train the model using Prefect pipeline"
	@echo "  serve             Start the web service locally"
	@echo "  deploy            Deploy services using Docker Compose"
	@echo "  monitor           Run monitoring pipeline"
	@echo "  clean             Clean up generated files"
	@echo "  build             Build Docker image"

# Install dependencies
install:
	pip install -r requirements.txt

# Setup development environment
setup-env:
	pip install pre-commit
	pre-commit install
	mkdir -p models logs monitoring Data/processed
	@echo "Development environment setup complete"

# Code quality checks
quality-checks:
	black --check src/ tests/
	isort --check-only src/ tests/
	flake8 src/ tests/

# Format code
format:
	black src/ tests/
	isort src/ tests/

# Run linting
lint:
	flake8 src/ tests/

# Run unit tests
test:
	pytest tests/ -v

# Run integration tests
integration-tests:
	pytest tests/ -v -m integration

# Train model using orchestration pipeline
train:
	python src/orchestration.py

# Start MLflow server
mlflow-server:
	mlflow server \
		--backend-store-uri sqlite:///mlflow.db \
		--default-artifact-root ./artifacts \
		--host 0.0.0.0 \
		--port 5000

# Start Prefect server (for local development)
prefect-server:
	prefect server start

# Serve model locally
serve:
	python src/web_service.py

# Run monitoring
monitor:
	python src/monitoring.py

# Build Docker image
build:
	docker build -t prostate-cancer-prediction .

# Deploy all services
deploy:
	docker-compose up -d

# Stop all services
stop:
	docker-compose down

# View logs
logs:
	docker-compose logs -f

# Clean up
clean:
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf *.egg-info/
	rm -rf build/
	rm -rf dist/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

# Setup complete MLOps environment
setup-mlops: setup-env install
	@echo "Installing additional MLOps tools..."
	pip install pre-commit
	pre-commit install
	@echo "Creating necessary directories..."
	mkdir -p models/{run_id,artifacts}
	mkdir -p logs
	mkdir -p monitoring/{reports,dashboards}
	mkdir -p Data/{raw,processed}
	@echo "MLOps environment setup complete!"

# Run complete CI pipeline locally
ci: quality-checks test
	@echo "CI pipeline completed successfully!"

# Development workflow
dev-setup: setup-mlops
	@echo "Starting MLflow server in background..."
	nohup make mlflow-server > mlflow.log 2>&1 &
	@echo "MLflow server started at http://localhost:5000"
	@echo "Development environment ready!"

# Production deployment
prod-deploy: build
	docker-compose -f docker-compose.yml up -d
	@echo "Production deployment started!"
	@echo "Services available at:"
	@echo "  - Web Service: http://localhost:9696"
	@echo "  - MLflow: http://localhost:5000"
	@echo "  - Grafana: http://localhost:3000"
	@echo "  - Adminer: http://localhost:8080"

# Health check all services
health-check:
	@echo "Checking service health..."
	curl -f http://localhost:9696/health || echo "Web service not responding"
	curl -f http://localhost:5000/health || echo "MLflow not responding"
	curl -f http://localhost:3000/api/health || echo "Grafana not responding"

# Generate sample predictions for testing monitoring
generate-test-data:
	python -c "
	import requests
	import json
	import time
	import random
	
	# Sample test data
	samples = [
		{'age': 45, 'bmi': 25.5, 'sleep_hours': 7.5, 'family_history': 'Yes', 'diet': 'Healthy', 'exercise_frequency': 'Daily', 'smoking_status': 'Never', 'alcohol_consumption': 'Light'},
		{'age': 60, 'bmi': 30.2, 'sleep_hours': 6.0, 'family_history': 'No', 'diet': 'Moderate', 'exercise_frequency': 'Weekly', 'smoking_status': 'Former', 'alcohol_consumption': 'None'},
		{'age': 35, 'bmi': 22.1, 'sleep_hours': 8.0, 'family_history': 'Yes', 'diet': 'Unhealthy', 'exercise_frequency': 'Rarely', 'smoking_status': 'Current', 'alcohol_consumption': 'Moderate'}
	]
	
	for i in range(10):
		sample = random.choice(samples)
		# Add some variation
		sample['age'] += random.randint(-5, 5)
		sample['bmi'] += random.uniform(-2, 2)
		
		try:
			response = requests.post('http://localhost:9696/predict_api', json=sample)
			print(f'Request {i+1}: {response.status_code}')
			time.sleep(1)
		except Exception as e:
			print(f'Error: {e}')
	"

# Database operations
db-migrate:
	@echo "Running database migrations..."
	# Add database migration commands here

# Backup models and data
backup:
	@echo "Creating backup..."
	tar -czf backup_$(shell date +%Y%m%d_%H%M%S).tar.gz models/ logs/ monitoring/
	@echo "Backup created successfully"

# Restore from backup
restore:
	@echo "Available backups:"
	ls -la backup_*.tar.gz
	@echo "To restore, run: tar -xzf backup_YYYYMMDD_HHMMSS.tar.gz"